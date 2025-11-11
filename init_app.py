#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, '/app/backend')

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path
import uuid

ROOT_DIR = Path('/app/backend')
load_dotenv(ROOT_DIR / '.env')

async def init_database():
    # Connect to MongoDB
    mongo_url = os.environ['MONGO_URL']
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ['DB_NAME']]
    
    print("Initializing database...")
    
    # Check if already initialized
    existing_cat = await db.categories.find_one({})
    if existing_cat:
        print("Database already initialized")
        return
    
    # Create categories
    categories = [
        {"id": str(uuid.uuid4()), "name_uk": "Електроніка", "name_ru": "Электроника", "name_en": "Electronics"},
        {"id": str(uuid.uuid4()), "name_uk": "Одяг", "name_ru": "Одежда", "name_en": "Clothing"},
        {"id": str(uuid.uuid4()), "name_uk": "Транспорт", "name_ru": "Транспорт", "name_en": "Transport"},
        {"id": str(uuid.uuid4()), "name_uk": "Нерухомість", "name_ru": "Недвижимость", "name_en": "Real Estate"},
        {"id": str(uuid.uuid4()), "name_uk": "Дім і сад", "name_ru": "Дом и сад", "name_en": "Home & Garden"},
        {"id": str(uuid.uuid4()), "name_uk": "Спорт", "name_ru": "Спорт", "name_en": "Sports"},
        {"id": str(uuid.uuid4()), "name_uk": "Дитячі товари", "name_ru": "Детские товары", "name_en": "Kids"},
        {"id": str(uuid.uuid4()), "name_uk": "Послуги", "name_ru": "Услуги", "name_en": "Services"},
        {"id": str(uuid.uuid4()), "name_uk": "Інше", "name_ru": "Другое", "name_en": "Other"}
    ]
    
    await db.categories.insert_many(categories)
    print(f"✓ Created {len(categories)} categories")
    
    # Create test admin user
    admin_user = {
        "id": str(uuid.uuid4()),
        "telegram_id": 1234567,
        "username": "admin",
        "first_name": "Admin",
        "last_name": "User",
        "avatar": None,
        "rating": 5.0,
        "ratings_count": 0,
        "is_pro": True,
        "pro_until": None,
        "is_admin": True,
        "is_banned": False,
        "banned_until": None,
        "ban_reason": None,
        "created_at": "2025-01-01T00:00:00"
    }
    
    await db.users.insert_one(admin_user)
    print("✓ Created admin user (telegram_id: 1234567)")
    
    print("\n✅ Database initialization complete!")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(init_database())