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
        'HTTP-Referer': process.env.APP_URL || 'https://sielcart-c2b0eahdauhpbtfa.malaysiawest-01.azurewebsites.net',
        'X-Title': 'Siel Cart E-Commerce Assistant',
    }
});

// 2. Initialize Database Connection Pool
const dbPool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT) || 3306,
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'siel_cart',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0,
    ssl: process.env.DB_HOST && process.env.DB_HOST.includes('aivencloud.com')
        ? { rejectUnauthorized: false }
        : false
});

const FALLBACK_MODELS = [
    'openrouter/free',
    'deepseek/deepseek-chat:free',
    'google/gemini-2.0-flash-exp:free'
];

const STANDARD_REFUSAL = "I can only assist with Siel Cart FAQs (how to order, returns/refunds, data handling), product recommendations, and order status inquiries. How may I help you today?";

/**
 * Static store rules and FAQ boundaries.
 */
const STORE_FACTS = `STORE FACTS (the only accurate description of how Siel Cart works):
SielCart is the online store of the UBAP Office at Central Luzon State University. It is pickup-only and cash-only. There is no delivery, no courier, and no online payment of any kind.

HOW TO ORDER:
1. Browse the Siel Cart catalog and open the product you want.
2. Choose the variation (such as size or color) if the product has one, set the quantity, then add it to your cart.
3. Open your cart and review the items. The whole cart is checked out together, so remove anything you are not buying yet. If an item is out of stock or the quantity is more than the remaining stock, checkout is blocked until you fix or remove that item.
4. Proceed to checkout. There is nothing to fill in. Your name and email come from your account, and the pickup location (UBAP Office) and payment method (Cash on Pickup) are fixed and shown for confirmation only.
5. Review your items and total, then place the order. The total is only the merchandise subtotal. There is no shipping fee, no tax, and no delivery charge.
6. You will receive a confirmation email with your order number, and the order starts as Pending.
7. The UBAP staff prepare the order. Once it is ready, you receive a second email with your claim number, your pickup date, and your pickup time slot.
8. Go to the UBAP Office within your time slot, present your claim number, and pay in cash when you receive your items. The order is then marked Completed and Paid.

PAYMENT: Cash on Pickup only, paid in person at the UBAP Office when the items are handed over. Amounts are in Philippine pesos. Siel Cart does not accept credit or debit cards, GCash, bank transfers, e-wallets, or any online or advance payment, and it does not store payment details.

CLAIM NUMBER: A claim number is issued only when the order becomes Ready for Pickup, not at checkout. Before that the order has an order number only. Present the claim number at the UBAP Office to collect the order.

PICKUP: Orders must be claimed within the assigned date and time slot. Unclaimed orders are cancelled. A customer who cannot come on the scheduled date should contact the UBAP Office by email or in person, preferably before the pickup period ends, and ask for the pickup to be rescheduled. The customer cannot change the date on the website. UBAP sets the new date and time slot, emails it to the customer, and shows it on the order details page together with the original schedule. The claim number stays the same. Someone else may collect on the customer's behalf as long as they bring the claim number of the order.

ORDER STATUS: A customer checks progress by logging in and opening My Orders, then the order. Statuses are Pending, Processing, Ready for Pickup, Completed, Cancelled, and the return statuses. There are no tracking numbers and no delivery updates because nothing is shipped.

CANCELLATION: A customer can cancel from the order details page only while the order is still Pending, choosing either change of mind or incorrect items. Once the order is being processed, they must contact the UBAP Office.

RETURNS AND REFUNDS: Returns, refunds, and exchanges cannot be requested through the website. For a defective or damaged item, or an item handed over in error, the customer contacts the UBAP Office directly by email or in person, and UBAP handles the verification and decision. Once UBAP has refunded or exchanged an item, the outcome is shown against that item on the customer's order details page.

FORBIDDEN CLAIMS: Never mention or ask for a shipping address, delivery address, shipping method, shipping fee, delivery date, courier, tracking number, tracking link, card payment, e-wallet, online payment, or cash on delivery. Never say an order will be shipped or delivered. If a customer asks about delivery or online payment, tell them plainly that Siel Cart is pickup and cash-on-pickup only, then explain the pickup process.`;

// Pre-filter non-e-commerce inputs (Math, Coding, Simple Off-Topic)
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

/**
 * Fetch available/in-stock products directly from database
 */
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

/**
 * Universal product pre-filtering across ALL categories and price constraints
 */
function getProductSuggestionsByQuery(userQuery, products) {
    const text = userQuery.toLowerCase();
    
    // Extract numerical target price (e.g., "200", "under 300 pesos", "below ₱500")
    const priceMatch = text.match(/(\d+)\s*(pesos|php|₱)?/i);
    const targetPrice = priceMatch ? parseFloat(priceMatch[1]) : null;

    // Standard product category keyword mappings
    const CATEGORIES = {
        apparel: ['shirt', 'tshirt', 't-shirt', 'hoodie', 'jacket', 'cap', 'hat', 'clothes', 'wear', 'apparel'],
        stationery: ['pen', 'ballpen', 'notebook', 'paper', 'pencil', 'pad', 'stationery', 'supplies', 'school'],
        accessories: ['lanyard', 'holder', 'id holder', 'keychain', 'badge', 'accessory', 'accessories'],
        bags: ['bag', 'tote', 'totebag', 'backpack', 'pouch'],
        drinkware: ['mug', 'tumbler', 'cup', 'bottle', 'flask', 'water bottle']
    };

    let filtered = products;

    // 1. Category Search: Match query against category synonyms or direct product name keywords
    let matchedCategoryItems = [];
    for (const [category, keywords] of Object.entries(CATEGORIES)) {
        if (keywords.some(kw => text.includes(kw))) {
            const matches = products.filter(p => 
                keywords.some(kw => p.name.toLowerCase().includes(kw))
            );
            matchedCategoryItems.push(...matches);
        }
    }

    // Deduplicate matches if category matches were found
    if (matchedCategoryItems.length > 0) {
        filtered = Array.from(new Set(matchedCategoryItems));
    }

    // 2. Budget Filtering: Apply price constraints if a number is present
    if (targetPrice) {
        if (text.includes('under') || text.includes('below') || text.includes('less than')) {
            const underItems = filtered.filter(p => p.price <= targetPrice);
            filtered = underItems.length > 0 ? underItems : filtered.filter(p => p.price <= targetPrice * 1.25);
        } else {
            const matching = filtered.filter(p => p.price <= targetPrice * 1.2);
            filtered = matching.length > 0 ? matching : filtered;
        }
    }

    // Default: Return up to 6 items if no specific filter reduced the catalog
    if (filtered.length === products.length) {
        filtered = products.slice(0, 6);
    }

    return filtered.map(p => `- \({p.name}: ₱\){p.price}`).join('\n');
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

        // STEP 1: Code-level check to instantly block math, coding, or trivia queries
        if (isIrrelevantQuery(message)) {
            return res.json({ response: STANDARD_REFUSAL });
        }

        // STEP 2: Fetch DB products (with hardcoded fallback if DB is down/empty)
        let dbProducts = [];
        try {
            dbProducts = await fetchAvailableProducts();
        } catch (dbErr) {
            console.error('Failed to fetch from DB:', dbErr);
        }

        if (!dbProducts || dbProducts.length === 0) {
            dbProducts = [
                { name: "UBAP Ballpen", price: 20 },
                { name: "CLSU Notebook", price: 50 },
                { name: "Siel Cart Lanyard", price: 80 },
                { name: "CLSU ID Holder", price: 100 },
                { name: "UBAP Mug", price: 200 },
                { name: "Siel Cart Tote Bag", price: 200 },
                { name: "CLSU Basic Shirt", price: 250 },
                { name: "CLSU Cap", price: 250 },
                { name: "CLSU T-Shirt", price: 350 },
                { name: "UBAP Hoodie", price: 750 }
            ];
        }

        // STEP 3: DIRECT PRODUCT INTERCEPTOR (Bypasses LLM disclaimers entirely)
        const msgLower = message.toLowerCase();
        const isRecommendationQuery = 
            msgLower.includes('suggest') || 
            msgLower.includes('recommend') || 
            msgLower.includes('product') || 
            msgLower.includes('item') || 
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
            msgLower.includes('pen');

        if (isRecommendationQuery) {
            const matchedList = getProductSuggestionsByQuery(message, dbProducts);
            
            // Immediately respond from Node.js in clean English without touching LLM
            return res.json({ 
                response: `Here are the available products matching your request:\n${matchedList}` 
            });
        }

        // STEP 4: If it's a general FAQ (e.g. "how to order", "where to pick up"), use the LLM
        const dynamicCatalog = dbProducts.map(p => `- \({p.name}: ₱\){p.price}`).join('\n');
        
        const systemInstruction = `CRITICAL ASSISTANT BOUNDARY:
You are strictly an e-commerce assistant for Siel Cart. You DO NOT answer math, coding, trivia, or off-topic queries.

LANGUAGE RULE:
Respond ONLY in English at all times.

AVAILABLE PRODUCT CATALOG IN OUR SHOP:
${dynamicCatalog}

${STORE_FACTS}

REFUSAL INSTRUCTIONS:
If the user query is unrelated to Siel Cart e-commerce, output EXACTLY this response in English:
"${STANDARD_REFUSAL}"

FORMATTING:
- Standard plain text only. No Markdown formatting.
- Use dashes (-) for lists.`;

        const responseText = await generateContentWithFallback(message, systemInstruction);
        return res.json({ response: responseText });

    } catch (error) {
        console.error('All models failed or server error occurred:', error);

        return res.status(200).json({ 
            response: "Here are some available products in our shop:\n- CLSU Notebook: ₱50\n- UBAP Mug: ₱200\n- Siel Cart Tote Bag: ₱200\n- CLSU Basic Shirt: ₱250\n- CLSU T-Shirt: ₱350" 
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
});