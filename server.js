import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import mysql from 'mysql2/promise';

const app = express();
app.use(cors());
app.use(express.json());

// 1. Initialize OpenRouter / OpenAI Client
const deepseek = new OpenAI({
    baseURL: 'https://openrouter.ai/api/v1',
    apiKey: process.env.OPENROUTER_API_KEY,
    defaultHeaders: {
        'HTTP-Referer': 'http://localhost:3000',
        'X-Title': 'Siel Cart E-Commerce Assistant',
    }
});

// 2. Initialize Database Connection Pool
const dbPool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'siel_cart',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

const FALLBACK_MODELS = [
    'openrouter/free',
    'deepseek/deepseek-chat:free',
    'google/gemini-2.0-flash-exp:free'
];

const STANDARD_REFUSAL = "I can only assist with Siel Cart FAQs (how to order, returns/refunds, data handling), product recommendations, and order status inquiries. How may I help you today?";

const FRIENDLY_ERROR_MESSAGE = "Our assistant is temporarily unavailable. Please browse our catalog on the store page or contact the UBAP Office directly for immediate assistance.";

/**
 * Static store rules - Condensed to force short outputs
 */
const STORE_FACTS = `STORE FACTS (Siel Cart - UBAP Office at CLSU):
Siel Cart is pickup-only and cash-only at the UBAP Office. No delivery, no couriers, no cards/GCash/online payments.

HOW TO ORDER:
To place an order:
1. Browse our catalog and select an item.
2. Choose your preferred size or variant, then add it to your cart.
3. Open your cart and review your items.
4. Proceed to checkout to confirm your order details.
5. Receive your claim number via email, then collect and pay in cash at the UBAP Office.

PICKUP & CANCELLATION:
- Claim Numbers are issued ONLY when status is "Ready for Pickup".
- Unclaimed orders are cancelled. To reschedule pickup, contact UBAP Office.
- Cancel orders on "My Orders" page ONLY while status is "Pending".

RETURNS & PRIVACY:
- Returns/refunds cannot be requested on the website. Contact UBAP Office directly for defective items.
- Data Privacy: [Privacy Policy](/privacy-policy)
- Terms & Conditions: [Terms & Conditions](/terms-and-conditions)`;

function isIrrelevantQuery(text) {
    const query = text.trim().toLowerCase();

    const GREETINGS = ['hi', 'hello', 'halu', 'hey', 'good morning', 'good afternoon', 'good evening', 'kumusta', 'yo'];
    if (GREETINGS.some(g => query === g || query.startsWith(g + ' '))) {
        return false;
    }

    const mathPattern = /^(\d+[\s\+\-\*\/\^%\=]+\d+|\b(what is|calculate|compute|solve)\b.*?\d+)/i;
    if (mathPattern.test(query)) return true;
    if (/^\d+\s*[\+\-\*\/]\s*\d+/.test(query)) return true;
    if (/\b(write code|python|javascript|function|html|css|sql|script)\b/i.test(query)) return true;
    if (/^(who is|what is the capital|tell me a story|write a poem|sing|meaning of life)/i.test(query)) return true;

    return false;
}

async function fetchAvailableProducts() {
    try {
        const [rows] = await dbPool.query(
            'SELECT name, price FROM products WHERE is_active = 1 AND stock > 0'
        );
        return rows;
    } catch (dbError) {
        console.error('Database fetch error:', dbError.message);
        return [];
    }
}

function getProductSuggestionsByQuery(userQuery, products) {
    const text = userQuery.toLowerCase();
    
    const priceMatch = text.match(/(\d+)\s*(pesos|php|₱)?/i);
    const targetPrice = priceMatch ? parseFloat(priceMatch[1]) : null;

    const CATEGORIES = {
        apparel: ['shirt', 'tshirt', 't-shirt', 'hoodie', 'jacket', 'cap', 'hat', 'clothes', 'wear', 'apparel'],
        stationery: ['pen', 'ballpen', 'notebook', 'paper', 'pencil', 'pad', 'stationery', 'supplies', 'school'],
        accessories: ['lanyard', 'holder', 'id holder', 'keychain', 'badge', 'accessory', 'accessories'],
        bags: ['bag', 'tote', 'totebag', 'backpack', 'pouch'],
        drinkware: ['mug', 'tumbler', 'cup', 'bottle', 'flask', 'water bottle']
    };

    let filtered = products;

    let matchedCategoryItems = [];
    for (const [category, keywords] of Object.entries(CATEGORIES)) {
        if (keywords.some(kw => text.includes(kw))) {
            const matches = products.filter(p => 
                keywords.some(kw => p.name.toLowerCase().includes(kw))
            );
            matchedCategoryItems.push(...matches);
        }
    }

    if (matchedCategoryItems.length > 0) {
        filtered = Array.from(new Set(matchedCategoryItems));
    }

    if (targetPrice) {
        if (text.includes('under') || text.includes('below') || text.includes('less than')) {
            const underItems = filtered.filter(p => p.price <= targetPrice);
            filtered = underItems.length > 0 ? underItems : filtered.filter(p => p.price <= targetPrice * 1.25);
        } else {
            const matching = filtered.filter(p => p.price <= targetPrice * 1.2);
            filtered = matching.length > 0 ? matching : filtered;
        }
    }

    const topThree = filtered.slice(0, 3);

    return topThree.map(p => `- **\({p.name}**: ₱\){p.price}`).join('\n');
}

async function generateContentWithFallback(message, systemInstruction) {
    let lastError = null;

    for (const modelName of FALLBACK_MODELS) {
        try {
            const completion = await deepseek.chat.completions.create({
                model: modelName,
                temperature: 0.0,
                messages: [
                    { role: 'system', content: systemInstruction },
                    { role: 'user', content: message }
                ],
            });

            let text = completion.choices[0]?.message?.content;
            
            if (text) {
                text = text.replace(/^(user\s*safety:\s*safe|user:safe)\s*/i, '').trim();

                if (text.length > 0) {
                    return text;
                }
            }
            throw new Error(`Model [${modelName}] returned an empty text payload.`);
        } catch (error) {
            console.warn(`Model [\({modelName}] failed/rate-limited:\){error.message}. Trying next model...`);
            lastError = error;
        }
    }

    throw lastError || new Error("All fallback models failed.");
}

app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;

        if (!message) {
            return res.status(400).json({ error: 'Message is required.' });
        }

        const msgLower = message.toLowerCase();

        // DIRECT OVERRIDE FOR ORDERING FAQ (Bypasses LLM long responses)
        const isOrderingQuery = 
            msgLower.includes('how to order') || 
            msgLower.includes('how do i order') || 
            msgLower.includes('place an order') || 
            msgLower.includes('how to place an order');

        if (isOrderingQuery) {
            return res.json({
                response: `To place an order:

1. Browse our catalog and select an item.
2. Choose your preferred size or variant, then add it to your cart.
3. Open your cart and review your items.
4. Proceed to checkout to confirm your order details.
5. Receive your claim number via email, then collect and pay in cash at the UBAP Office.`
            });
        }

        if (isIrrelevantQuery(message)) {
            return res.json({ response: STANDARD_REFUSAL });
        }

        let dbProducts = [];
        try {
            dbProducts = await fetchAvailableProducts();
        } catch (dbErr) {
            console.error('Failed to fetch from DB:', dbErr);
        }

        if (!dbProducts || dbProducts.length === 0) {
            dbProducts = [
                { name: "CLSU Notebook", price: 50 },
                { name: "Siel Cart Lanyard", price: 80 },
                { name: "UBAP Mug", price: 200 },
                { name: "Siel Cart Tote Bag", price: 200 },
                { name: "CLSU Basic Shirt", price: 250 },
                { name: "UBAP Hoodie", price: 750 }
            ];
        }

        const isFAQIntent = 
            msgLower.includes('how') || 
            msgLower.includes('where') || 
            msgLower.includes('when') || 
            msgLower.includes('can i') || 
            msgLower.includes('policy') || 
            msgLower.includes('status') || 
            msgLower.includes('cancel') || 
            msgLower.includes('return') || 
            msgLower.includes('refund') || 
            msgLower.includes('pay') || 
            msgLower.includes('privacy');

        const isRecommendationQuery = 
            !isFAQIntent && (
                msgLower.includes('suggest') || 
                msgLower.includes('recommend') || 
                msgLower.includes('price') || 
                msgLower.includes('pesos') || 
                msgLower.includes('php') || 
                msgLower.includes('under') || 
                msgLower.includes('below') || 
                msgLower.includes('shirt') || 
                msgLower.includes('apparel') || 
                msgLower.includes('mug') || 
                msgLower.includes('bag') || 
                msgLower.includes('notebook') || 
                msgLower.includes('pen')
            );

        if (isRecommendationQuery) {
            const matchedList = getProductSuggestionsByQuery(message, dbProducts);
            
            return res.json({ 
                response: `Here are 3 product recommendations matching your request:\n\n${matchedList}` 
            });
        }

        const dynamicCatalog = dbProducts.slice(0, 5).map(p => `- **\({p.name}**: ₱\){p.price}`).join('\n');
        
        const systemInstruction = `CRITICAL ASSISTANT BOUNDARY:
You are strictly an e-commerce assistant for Siel Cart. You DO NOT answer math, coding, trivia, or off-topic queries.

LANGUAGE RULE:
Respond ONLY in English at all times.

STRICT LENGTH & FORMATTING RULES:
- Output ONLY short answers.
- When asked "how to order", reply EXACTLY with the 5 numbered steps listed in STORE FACTS under "HOW TO ORDER".
- DO NOT add extra commentary, detailed explanations, or closing questions like "Is there anything else I can help you with?".

AVAILABLE PRODUCT CATALOG IN OUR SHOP:
${dynamicCatalog}

${STORE_FACTS}

REFUSAL INSTRUCTIONS:
If the user query is unrelated to Siel Cart e-commerce, output EXACTLY this response in English:
"${STANDARD_REFUSAL}"`;

        const responseText = await generateContentWithFallback(message, systemInstruction);
        return res.json({ response: responseText });

    } catch (error) {
        console.error('All models failed or server error occurred:', error);

        return res.json({ 
            response: FRIENDLY_ERROR_MESSAGE 
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
});