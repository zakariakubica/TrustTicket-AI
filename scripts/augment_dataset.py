"""
Data Augmentation Script for TrustTicket
Expands the 30-sample dataset to 200+ samples using realistic variations

Author: Zakaria Kubica
Date: December 2025
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Real artists and events (popular concerts/sports)
LEGITIMATE_EVENTS = [
    # Pop/Rock
    ("Taylor Swift", "The Eras Tour", ["Wembley Stadium", "O2 Arena", "Manchester Arena"]),
    ("Ed Sheeran", "Mathematics Tour", ["Manchester Arena", "Birmingham Arena", "Glasgow OVO Hydro"]),
    ("Coldplay", "Music of the Spheres", ["Wembley Stadium", "Cardiff Stadium", "Hampden Park"]),
    ("Beyoncé", "Renaissance World Tour", ["Tottenham Hotspur Stadium", "Manchester Arena"]),
    ("Harry Styles", "Love On Tour", ["The O2", "Manchester Arena", "Birmingham Arena"]),
    ("Adele", "Weekends with Adele", ["Hyde Park", "O2 Arena"]),
    ("The Weeknd", "After Hours Til Dawn", ["Wembley Stadium", "Manchester Arena"]),
    ("Arctic Monkeys", "The Car Tour", ["Emirates Stadium", "Manchester Arena"]),
    ("Sam Smith", "Gloria The Tour", ["O2 Arena", "Manchester Arena"]),
    ("Lewis Capaldi", "Broken By Desire Tour", ["Glasgow OVO Hydro", "Manchester Arena"]),
    
    # Hip Hop/R&B
    ("Drake", "It's All A Blur Tour", ["O2 Arena", "Manchester Arena"]),
    ("Travis Scott", "Utopia Tour", ["O2 Arena", "Wembley Arena"]),
    ("SZA", "SOS Tour", ["O2 Arena", "Birmingham Arena"]),
    
    # Electronic/Dance
    ("Swedish House Mafia", "Paradise Again", ["O2 Arena", "Manchester Arena"]),
    ("Calvin Harris", "Funk Wav Bounces", ["Ushuaïa Ibiza", "O2 Arena"]),
    
    # Sports Events
    ("Premier League", "Manchester United vs Liverpool", ["Old Trafford"]),
    ("Premier League", "Arsenal vs Chelsea", ["Emirates Stadium"]),
    ("Premier League", "Manchester City vs Tottenham", ["Etihad Stadium"]),
    ("Champions League", "Final 2025", ["Wembley Stadium"]),
    ("Rugby World Cup", "England vs New Zealand", ["Twickenham Stadium"]),
    ("Wimbledon", "Men's Final", ["Centre Court"]),
    ("F1", "British Grand Prix", ["Silverstone"]),
]

# Legitimate platforms with trust scores
LEGITIMATE_PLATFORMS = [
    ("Ticketmaster", 0.95),
    ("See Tickets", 0.92),
    ("AXS", 0.90),
    ("StubHub", 0.85),
    ("Viagogo", 0.75),
    ("Eventim", 0.88),
    ("Gigantic", 0.87),
    ("TicketWeb", 0.86),
]

# Scam platforms
SCAM_PLATFORMS = [
    ("Facebook Marketplace", 0.35),
    ("Gumtree", 0.40),
    ("Craigslist UK", 0.30),
    ("Instagram DM", 0.15),
    ("WhatsApp Group", 0.20),
    ("Random Website", 0.25),
    ("Email Offer", 0.18),
]

# Scam red flags and descriptions
SCAM_PATTERNS = [
    {
        "red_flags": "urgency_language,extremely_low_price",
        "description_templates": [
            "URGENT SALE!!! Must sell TODAY! Price WAY below market. Contact me ASAP!!!",
            "LAST CHANCE! Selling cheap because I can't go anymore! Hurry before gone!!!",
            "DESPERATE SALE - Need money NOW! Amazing deal, won't last long!!!",
            "LIMITED TIME OFFER! Price slashed! Act fast or miss out!!!"
        ],
        "price_multiplier": 0.3,  # 30% of market price
        "payment_method": "Bank transfer only",
        "contact_method": "WhatsApp/Telegram only"
    },
    {
        "red_flags": "crypto_payment,no_verification,suspicious_contact",
        "description_templates": [
            "Genuine tickets available. Bitcoin/crypto payment preferred. Contact off-platform for details.",
            "Premium seats, must sell. Accept cryptocurrency only. DM for payment info.",
            "VIP tickets, crypto payments accepted. Fast transaction, no middleman.",
            "Tickets available, prefer crypto payment for faster processing."
        ],
        "price_multiplier": 0.7,
        "payment_method": "Cryptocurrency only",
        "contact_method": "Telegram/Discord"
    },
    {
        "red_flags": "untraceable_payment,no_seller_history,poor_grammar",
        "description_templates": [
            "great ticket for sale need cash fast no refund sorry contact me quick",
            "selling ticket cheap price good deal message me now for buy",
            "ticket available urgent need sell today contact fast",
            "have ticket must sell quick cash only no question ask"
        ],
        "price_multiplier": 0.5,
        "payment_method": "Cash/Gift cards",
        "contact_method": "Unknown seller"
    },
    {
        "red_flags": "extreme_price_gouging,unverified_source",
        "description_templates": [
            "Exclusive VIP tickets with backstage access! Rare opportunity! Premium price for premium experience!",
            "Front row seats guaranteed! Official looking tickets. Price reflects exclusivity.",
            "Platinum tickets with meet & greet included! Investment opportunity!",
            "Ultra VIP package with private entrance! Once in a lifetime chance!"
        ],
        "price_multiplier": 3.5,  # 350% markup
        "payment_method": "Wire transfer",
        "contact_method": "Personal email"
    }
]

# Legitimate descriptions
LEGITIMATE_DESCRIPTIONS = [
    "Official resale ticket. Verified by platform. Can't attend due to schedule conflict.",
    "Genuine ticket purchased from Ticketmaster. Selling at face value. Digital transfer available.",
    "Authentic ticket with proof of purchase. Platform verified seller. Fast delivery.",
    "Official ticket from authorized vendor. Secure transfer through platform. Buyer protection included.",
    "Face value resale. Verified seller with excellent rating. Platform guaranteed.",
    "Legitimate ticket with original receipt. Secure payment through official channels.",
    "Authorized resale with full refund protection. Verified by platform security.",
    "Official ticket transfer. Platform managed transaction. Buyer fully protected."
]

def generate_market_price(event_type):
    """Generate realistic market prices based on event type"""
    if "Taylor Swift" in event_type or "Beyoncé" in event_type:
        return random.randint(120, 350)
    elif "Stadium" in event_type or "Wembley" in event_type:
        return random.randint(80, 250)
    elif "Premier League" in event_type:
        return random.randint(45, 180)
    elif "Champions League" in event_type or "World Cup" in event_type:
        return random.randint(100, 400)
    elif "Arena" in event_type:
        return random.randint(50, 150)
    else:
        return random.randint(60, 200)

def generate_legitimate_listing(listing_id):
    """Generate a legitimate ticket listing"""
    artist, event, venues = random.choice(LEGITIMATE_EVENTS)
    venue = random.choice(venues)
    platform, trust_score = random.choice(LEGITIMATE_PLATFORMS)
    
    market_price = generate_market_price(f"{artist} {venue}")
    price = random.randint(int(market_price * 0.9), int(market_price * 1.3))  # 90-130% of market
    
    event_date = datetime.now() + timedelta(days=random.randint(30, 180))
    listing_date = event_date - timedelta(days=random.randint(14, 90))
    days_until = (event_date - datetime.now()).days
    
    sections = ["General Admission", "Section A", "Section B", "Lower Tier", "Upper Tier", "Standing"]
    rows = [None, "A", "B", "C", "D", "1", "2", "3", "4", "5"]
    seats = [None, "12", "15", "20", "8", "10"]
    
    seller_rating = round(random.uniform(4.0, 5.0), 1)
    
    return {
        "listing_id": listing_id,
        "event": event,
        "artist_performer": artist,
        "venue": venue,
        "section": random.choice(sections),
        "row": random.choice(rows),
        "seat": random.choice(seats),
        "price_gbp": price,
        "currency": "GBP",
        "description": random.choice(LEGITIMATE_DESCRIPTIONS),
        "seller_name": f"User{random.randint(1000, 9999)}",
        "seller_platform": platform,
        "contact_method": "Platform messaging",
        "payment_method": "Platform payment system",
        "listing_date": listing_date.strftime("%Y-%m-%d"),
        "event_date": event_date.strftime("%Y-%m-%d"),
        "days_until_event": days_until,
        "is_scam": 0,
        "scam_confidence": round(random.uniform(0.05, 0.20), 2),
        "platform_trust_score": trust_score,
        "weight": round(random.uniform(2.5, 4.5), 1),
        "red_flags": None
    }

def generate_scam_listing(listing_id):
    """Generate a scam ticket listing"""
    artist, event, venues = random.choice(LEGITIMATE_EVENTS)
    venue = random.choice(venues)
    platform, trust_score = random.choice(SCAM_PLATFORMS)
    scam_pattern = random.choice(SCAM_PATTERNS)
    
    market_price = generate_market_price(f"{artist} {venue}")
    price = int(market_price * scam_pattern["price_multiplier"])
    
    event_date = datetime.now() + timedelta(days=random.randint(5, 45))  # Shorter time = more urgency
    listing_date = event_date - timedelta(days=random.randint(1, 14))  # Recently posted
    days_until = (event_date - datetime.now()).days
    
    sections = ["VIP", "Front Row", "Best Seats", "Premium", None]
    
    return {
        "listing_id": listing_id,
        "event": event,
        "artist_performer": artist,
        "venue": venue,
        "section": random.choice(sections),
        "row": None,
        "seat": None,
        "price_gbp": price,
        "currency": "GBP",
        "description": random.choice(scam_pattern["description_templates"]),
        "seller_name": None if random.random() > 0.3 else f"NewUser{random.randint(1, 999)}",
        "seller_platform": platform,
        "contact_method": scam_pattern["contact_method"],
        "payment_method": scam_pattern["payment_method"],
        "listing_date": listing_date.strftime("%Y-%m-%d"),
        "event_date": event_date.strftime("%Y-%m-%d"),
        "days_until_event": days_until,
        "is_scam": 1,
        "scam_confidence": round(random.uniform(0.70, 0.95), 2),
        "platform_trust_score": trust_score,
        "weight": round(random.uniform(0.5, 1.5), 1),
        "red_flags": scam_pattern["red_flags"]
    }

def augment_dataset(original_csv_path, output_csv_path, target_total=200):
    """
    Augment the dataset from 30 to target_total samples
    Maintains realistic legitimate:scam ratio (70:30)
    """
    print("="*60)
    print("🔄 Data Augmentation - TrustTicket Dataset Expansion")
    print("="*60)
    
    # Load original dataset
    print(f"\n📊 Loading original dataset from: {original_csv_path}")
    original_df = pd.read_csv(original_csv_path)
    print(f"Original dataset size: {len(original_df)} samples")
    print(f"Original distribution:\n{original_df['is_scam'].value_counts()}")
    
    # Calculate how many new samples to generate
    num_to_generate = target_total - len(original_df)
    num_legitimate = int(num_to_generate * 0.70)  # 70% legitimate
    num_scam = num_to_generate - num_legitimate    # 30% scam
    
    print(f"\n🎯 Target: {target_total} total samples")
    print(f"Generating {num_to_generate} new samples:")
    print(f"  - {num_legitimate} legitimate listings")
    print(f"  - {num_scam} scam listings")
    
    # Generate new listings
    new_listings = []
    current_id = len(original_df) + 1
    
    print("\n⚙️  Generating legitimate listings...")
    for i in range(num_legitimate):
        listing = generate_legitimate_listing(f"TL{current_id:03d}")
        new_listings.append(listing)
        current_id += 1
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_legitimate} legitimate listings...")
    
    print("\n⚙️  Generating scam listings...")
    for i in range(num_scam):
        listing = generate_scam_listing(f"TL{current_id:03d}")
        new_listings.append(listing)
        current_id += 1
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_scam} scam listings...")
    
    # Combine with original data
    new_df = pd.DataFrame(new_listings)
    augmented_df = pd.concat([original_df, new_df], ignore_index=True)
    
    # Shuffle the dataset
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save augmented dataset
    augmented_df.to_csv(output_csv_path, index=False)
    
    print("\n" + "="*60)
    print("✅ Data Augmentation Complete!")
    print("="*60)
    print(f"📁 Saved to: {output_csv_path}")
    print(f"📊 Total samples: {len(augmented_df)}")
    print(f"\n📈 Final distribution:")
    print(augmented_df['is_scam'].value_counts())
    print(f"\nLegitimate: {(augmented_df['is_scam'] == 0).sum()} ({(augmented_df['is_scam'] == 0).sum() / len(augmented_df) * 100:.1f}%)")
    print(f"Scam:       {(augmented_df['is_scam'] == 1).sum()} ({(augmented_df['is_scam'] == 1).sum() / len(augmented_df) * 100:.1f}%)")
    
    # Show sample of new data
    print("\n📋 Sample of augmented data:")
    print(augmented_df[['listing_id', 'artist_performer', 'event', 'price_gbp', 'is_scam']].head(10))
    
    return augmented_df

if __name__ == "__main__":
    # Paths
    original_csv = "data/raw/ticket_listings_dataset.csv"
    output_csv = "data/raw/ticket_listings_dataset_augmented.csv"
    
    # Check if original file exists
    if not os.path.exists(original_csv):
        print(f"❌ Error: Original dataset not found at {original_csv}")
        print("Please ensure the dataset is in the correct location.")
        exit(1)
    
    # Run augmentation
    augmented_df = augment_dataset(
        original_csv_path=original_csv,
        output_csv_path=output_csv,
        target_total=200  # Change this to generate more/fewer samples
    )
    
    print("\n🎉 Dataset ready for training!")
    print(f"Use this file for training: {output_csv}")
