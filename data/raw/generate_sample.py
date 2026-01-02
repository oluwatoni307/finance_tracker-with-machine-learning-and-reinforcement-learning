import pandas as pd
import random

# Set seed for reproducibility
random.seed(42)

def generate_variations(base_merchants, category, variations_per_merchant=8):
    """Generate realistic transaction description variations"""
    
    # Common suffixes/modifiers by category
    location_suffixes = [
        "store #1234", "store #5678", "location 001", "branch 42",
        "downtown", "mall location", "main street", "plaza",
        "shopping center", "market", "north", "south", "east", "west"
    ]
    
    online_modifiers = [
        "online order", "online purchase", "web order", "mobile app",
        "app purchase", "digital order", "online", "internet order"
    ]
    
    transaction_ids = [
        "trans 12345", "order 98765", "purchase 4567", "ref 789",
        "transaction id", "conf 111", "receipt 222"
    ]
    
    payment_methods = [
        "card payment", "credit card", "debit card", "payment",
        "charge", "purchase", "sale"
    ]
    
    time_modifiers = [
        "early morning", "morning", "afternoon", "evening", "late night",
        "monday", "weekend", "today", "yesterday"
    ]
    
    # Category-specific modifiers
    category_modifiers = {
        'groceries': ['grocery pickup', 'delivery', 'curbside', 'express', 'fresh', 'organic section'],
        'dining': ['drive thru', 'takeout', 'delivery', 'dine in', 'to go', 'pickup order'],
        'transport': ['ride', 'trip', 'fare', 'service', 'rental', 'charge'],
        'entertainment': ['subscription', 'monthly', 'annual', 'premium', 'standard plan'],
        'utilities': ['bill payment', 'monthly bill', 'service', 'account', 'autopay'],
        'shopping': ['purchase', 'order', 'item', 'product', 'sale'],
        'health': ['appointment', 'visit', 'copay', 'service', 'prescription', 'refill'],
        'fitness': ['membership', 'monthly dues', 'class', 'session', 'fee'],
        'travel': ['booking', 'reservation', 'stay', 'night', 'trip', 'fare'],
        'personal care': ['appointment', 'service', 'treatment', 'session'],
        'pets': ['service', 'appointment', 'purchase', 'order'],
        'other': ['payment', 'fee', 'charge', 'bill', 'service']
    }
    
    variations = []
    
    for merchant in base_merchants:
        # Always include the base merchant name
        variations.append(merchant)
        
        # Generate variations
        count = 1
        attempts = 0
        max_attempts = variations_per_merchant * 3
        
        while count < variations_per_merchant and attempts < max_attempts:
            attempts += 1
            variation_type = random.choice(['location', 'online', 'id', 'payment', 'category', 'time', 'compound'])
            
            if variation_type == 'location':
                var = f"{merchant} {random.choice(location_suffixes)}"
            elif variation_type == 'online':
                var = f"{merchant} {random.choice(online_modifiers)}"
            elif variation_type == 'id':
                var = f"{merchant} {random.choice(transaction_ids)}"
            elif variation_type == 'payment':
                var = f"{merchant} {random.choice(payment_methods)}"
            elif variation_type == 'category' and category in category_modifiers:
                var = f"{merchant} {random.choice(category_modifiers[category])}"
            elif variation_type == 'time':
                var = f"{merchant} {random.choice(time_modifiers)}"
            elif variation_type == 'compound':
                # Combine multiple modifiers
                mods = []
                if random.random() > 0.5 and category in category_modifiers:
                    mods.append(random.choice(category_modifiers[category]))
                if random.random() > 0.6:
                    mods.append(random.choice(location_suffixes))
                if random.random() > 0.7:
                    mods.append(random.choice(transaction_ids))
                if mods:
                    var = f"{merchant} {' '.join(mods)}"
                else:
                    continue
            else:
                continue
            
            # Avoid exact duplicates
            if var not in variations and var != merchant:
                variations.append(var)
                count += 1
    
    return variations

# Base merchants per category (representative samples)
base_data = {
    'groceries': [
        'walmart', 'target', 'whole foods', 'kroger', 'safeway', 'costco', 'aldi',
        'trader joes', 'publix', 'wegmans', 'food lion', 'sprouts', 'instacart',
        'heb', 'giant eagle', 'albertsons', 'shoprite', 'harris teeter', 'meijer',
        'fresh market'
    ],
    'dining': [
        'mcdonalds', 'starbucks', 'chipotle', 'subway', 'taco bell', 'burger king',
        'wendys', 'dunkin', 'panera bread', 'chick fil a', 'pizza hut', 'dominos',
        'olive garden', 'applebees', 'five guys', 'shake shack', 'kfc', 'popeyes',
        'doordash', 'uber eats', 'grubhub', 'chilis', 'panda express', 'sonic',
        'in n out'
    ],
    'transport': [
        'uber', 'lyft', 'shell', 'chevron', 'exxon', 'bp', 'mobil', 'circle k',
        'speedway', 'taxi', 'parking', 'toll', 'car wash', 'auto repair',
        'enterprise', 'hertz', 'budget', 'jiffy lube', 'autozone', 'bus pass',
        'metro', 'train ticket', 'amtrak', 'greyhound'
    ],
    'entertainment': [
        'netflix', 'spotify', 'hulu', 'disney plus', 'amazon prime', 'hbo max',
        'youtube premium', 'apple tv', 'amc theatres', 'regal', 'xbox', 'playstation',
        'steam', 'twitch', 'audible', 'paramount plus', 'peacock', 'concert ticket',
        'ticketmaster', 'stubhub', 'six flags', 'bowling', 'arcade', 'escape room'
    ],
    'utilities': [
        'electric bill', 'water bill', 'gas bill', 'internet bill', 'phone bill',
        'verizon', 'att', 'tmobile', 'comcast', 'spectrum', 'xfinity', 'cox',
        'waste management', 'trash service', 'centurylink', 'frontier', 'directv',
        'dish network', 'cricket wireless', 'boost mobile'
    ],
    'shopping': [
        'amazon', 'ebay', 'etsy', 'target', 'walmart', 'best buy', 'apple store',
        'costco', 'kohls', 'macys', 'nordstrom', 'tj maxx', 'home depot', 'lowes',
        'ikea', 'wayfair', 'shein', 'nike', 'adidas', 'old navy', 'gap',
        'michaels', 'hobby lobby', 'petco', 'petsmart', 'dollar tree', 'ross',
        'marshalls', 'sephora', 'ulta'
    ],
    'health': [
        'cvs pharmacy', 'walgreens', 'rite aid', 'doctor', 'dentist', 'urgent care',
        'hospital', 'clinic', 'optometrist', 'physical therapy', 'quest diagnostics',
        'labcorp', 'minute clinic', 'dermatologist', 'cardiologist', 'chiropractor',
        'pharmacy', 'prescription', 'medical center', 'health clinic'
    ],
    'fitness': [
        'planet fitness', 'la fitness', '24 hour fitness', 'lifetime fitness',
        'golds gym', 'anytime fitness', 'crunch', 'equinox', 'ymca', 'crossfit',
        'orange theory', 'soulcycle', 'yoga studio', 'pilates', 'personal trainer',
        'peloton', 'gym membership', 'boxing gym', 'spin class', 'gnc'
    ],
    'travel': [
        'marriott', 'hilton', 'hyatt', 'holiday inn', 'best western', 'airbnb',
        'vrbo', 'expedia', 'booking com', 'hotels com', 'delta airlines',
        'american airlines', 'united airlines', 'southwest', 'jetblue', 'spirit airlines',
        'carnival cruise', 'royal caribbean', 'hotel', 'motel', 'resort'
    ],
    'personal care': [
        'haircut', 'hair salon', 'barber', 'supercuts', 'great clips', 'nail salon',
        'spa', 'massage', 'facial', 'manicure', 'pedicure', 'dry cleaning',
        'laundry service', 'waxing', 'tanning salon', 'tattoo', 'salon', 'barber shop'
    ],
    'pets': [
        'veterinarian', 'vet clinic', 'pet store', 'petco', 'petsmart', 'chewy',
        'pet grooming', 'dog grooming', 'pet boarding', 'dog walker', 'pet sitting',
        'pet daycare', 'dog training', 'vet visit', 'animal hospital', 'banfield'
    ],
    'other': [
        'bank fee', 'atm', 'insurance', 'loan payment', 'mortgage', 'rent payment',
        'childcare', 'daycare', 'tuition', 'donation', 'subscription fee', 'legal fee',
        'tax payment', 'parking fine', 'traffic ticket', 'wells fargo', 'chase bank',
        'venmo', 'paypal', 'cash app', 'storage unit', 'uhaul', 'church donation',
        'attorney', 'accountant', 'fedex', 'ups', 'usps', 'postage'
    ]
}

# Generate balanced dataset
all_data = []
target_per_category = 200  # Target 200 examples per category

for category, merchants in base_data.items():
    variations_needed = target_per_category // len(merchants) + 1
    variations = generate_variations(merchants, category, variations_needed)
    
    # Limit to target
    variations = variations[:target_per_category]
    
    for variation in variations:
        all_data.append({
            'Description': variation,
            'Category': category
        })

# Create DataFrame
df = pd.DataFrame(all_data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('labeled_transactions_balanced.csv', index=False)

print(f"✅ Generated {len(df)} training examples")
print(f"\n📊 Examples per category:")
print(df['Category'].value_counts().sort_index())
print(f"\n💾 Saved to: labeled_transactions_balanced.csv")
print(f"\nSample rows:")
print(df.head(20))