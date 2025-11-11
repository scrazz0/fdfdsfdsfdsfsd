from fastapi import FastAPI, APIRouter, HTTPException, File, UploadFile, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
from PIL import Image
import io
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create uploads directory
UPLOADS_DIR = ROOT_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI()
api_router = APIRouter(prefix="/api")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except:
                self.disconnect(user_id)

manager = ConnectionManager()

# Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    telegram_id: Optional[int] = None
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar: Optional[str] = None
    rating: float = 0.0
    ratings_count: int = 0
    is_pro: bool = False
    pro_until: Optional[str] = None
    is_admin: bool = False
    is_banned: bool = False
    banned_until: Optional[str] = None
    ban_reason: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Category(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name_uk: str
    name_ru: str
    name_en: str
    icon: Optional[str] = None

class Listing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: str
    price: float
    currency: str  # UAH, USD, EUR
    city: str
    category_id: str
    images: List[str] = []
    contact_telegram: Optional[str] = None
    contact_phone: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected
    is_pro: bool = False
    views: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Chat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    listing_id: str
    buyer_id: str
    seller_id: str
    last_message: Optional[str] = None
    last_message_at: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: str
    sender_id: str
    text: str
    is_read: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Rating(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    rated_by: str
    stars: int  # 1-5
    comment: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Ban(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    banned_by: str
    reason: str
    until: Optional[str] = None  # None = permanent
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# Create requests
class UserCreate(BaseModel):
    telegram_id: Optional[int] = None
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class ListingCreate(BaseModel):
    user_id: str
    title: str
    description: str
    price: float
    currency: str
    city: str
    category_id: str
    contact_telegram: Optional[str] = None
    contact_phone: Optional[str] = None

class MessageCreate(BaseModel):
    chat_id: str
    sender_id: str
    text: str

class RatingCreate(BaseModel):
    user_id: str
    rated_by: str
    stars: int
    comment: Optional[str] = None

class BanCreate(BaseModel):
    user_id: str
    banned_by: str
    reason: str
    until: Optional[str] = None

class ProPurchase(BaseModel):
    user_id: str
    telegram_payment_id: str

# Routes
@api_router.get("/")
async def root():
    return {"message": "СКР Барахолка API"}

# Users
@api_router.post("/users", response_model=User)
async def create_user(input: UserCreate):
    # Check if user exists
    existing = await db.users.find_one({"telegram_id": input.telegram_id}, {"_id": 0})
    if existing:
        return User(**existing)
    
    user = User(**input.model_dump())
    doc = user.model_dump()
    await db.users.insert_one(doc)
    return user

@api_router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user)

@api_router.get("/users/telegram/{telegram_id}", response_model=User)
async def get_user_by_telegram(telegram_id: int):
    user = await db.users.find_one({"telegram_id": telegram_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user)

@api_router.put("/users/{user_id}/avatar")
async def update_avatar(user_id: str, file: UploadFile = File(...)):
    # Read and optimize image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img.thumbnail((400, 400))
    
    # Save image
    filename = f"avatar_{user_id}_{uuid.uuid4()}.jpg"
    filepath = UPLOADS_DIR / filename
    img.save(filepath, "JPEG", quality=85)
    
    # Update user
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"avatar": filename}}
    )
    
    return {"avatar": filename}

@api_router.post("/users/{user_id}/pro")
async def purchase_pro(user_id: str, payment: ProPurchase):
    user = await db.users.find_one({"id": user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Add 30 days PRO
    pro_until = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"is_pro": True, "pro_until": pro_until}}
    )
    
    return {"is_pro": True, "pro_until": pro_until}

# Categories
@api_router.get("/categories", response_model=List[Category])
async def get_categories():
    categories = await db.categories.find({}, {"_id": 0}).to_list(100)
    return [Category(**cat) for cat in categories]

@api_router.post("/categories", response_model=Category)
async def create_category(category: Category):
    doc = category.model_dump()
    await db.categories.insert_one(doc)
    return category

# Listings
@api_router.post("/listings", response_model=Listing)
async def create_listing(listing: ListingCreate):
    user = await db.users.find_one({"id": listing.user_id}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_listing = Listing(**listing.model_dump())
    
    # If user has PRO, mark listing as PRO
    if user.get("is_pro"):
        new_listing.is_pro = True
    
    doc = new_listing.model_dump()
    await db.listings.insert_one(doc)
    
    # Update stats
    await db.stats.update_one(
        {"id": "global"},
        {"$inc": {"total_listings": 1}},
        upsert=True
    )
    
    return new_listing

@api_router.get("/listings", response_model=List[Listing])
async def get_listings(
    status: Optional[str] = "approved",
    category_id: Optional[str] = None,
    city: Optional[str] = None,
    search: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    currency: Optional[str] = None,
    skip: int = 0,
    limit: int = 20
):
    query = {}
    if status:
        query["status"] = status
    if category_id:
        query["category_id"] = category_id
    if city:
        query["city"] = city
    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"description": {"$regex": search, "$options": "i"}}
        ]
    if min_price is not None or max_price is not None:
        query["price"] = {}
        if min_price is not None:
            query["price"]["$gte"] = min_price
        if max_price is not None:
            query["price"]["$lte"] = max_price
    if currency:
        query["currency"] = currency
    
    listings = await db.listings.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    return [Listing(**listing) for listing in listings]

@api_router.get("/listings/{listing_id}", response_model=Listing)
async def get_listing(listing_id: str):
    listing = await db.listings.find_one({"id": listing_id}, {"_id": 0})
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Increment views
    await db.listings.update_one(
        {"id": listing_id},
        {"$inc": {"views": 1}}
    )
    listing["views"] = listing.get("views", 0) + 1
    
    return Listing(**listing)

@api_router.post("/listings/{listing_id}/images")
async def upload_listing_image(listing_id: str, file: UploadFile = File(...)):
    # Read and optimize image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img.thumbnail((1200, 1200))
    
    # Save image
    filename = f"listing_{listing_id}_{uuid.uuid4()}.jpg"
    filepath = UPLOADS_DIR / filename
    img.save(filepath, "JPEG", quality=85)
    
    # Update listing
    await db.listings.update_one(
        {"id": listing_id},
        {"$push": {"images": filename}}
    )
    
    return {"image": filename}

@api_router.put("/listings/{listing_id}/moderate")
async def moderate_listing(listing_id: str, status: str, admin_id: str):
    # Check if admin
    admin = await db.users.find_one({"id": admin_id}, {"_id": 0})
    if not admin or not admin.get("is_admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.listings.update_one(
        {"id": listing_id},
        {"$set": {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    
    return {"status": status}

# Chats
@api_router.post("/chats", response_model=Chat)
async def create_chat(listing_id: str, buyer_id: str):
    listing = await db.listings.find_one({"id": listing_id}, {"_id": 0})
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Check if chat already exists
    existing = await db.chats.find_one({
        "listing_id": listing_id,
        "buyer_id": buyer_id
    }, {"_id": 0})
    
    if existing:
        return Chat(**existing)
    
    chat = Chat(
        listing_id=listing_id,
        buyer_id=buyer_id,
        seller_id=listing["user_id"]
    )
    
    doc = chat.model_dump()
    await db.chats.insert_one(doc)
    return chat

@api_router.get("/chats/user/{user_id}", response_model=List[Chat])
async def get_user_chats(user_id: str):
    chats = await db.chats.find({
        "$or": [{"buyer_id": user_id}, {"seller_id": user_id}]
    }, {"_id": 0}).sort("last_message_at", -1).to_list(100)
    return [Chat(**chat) for chat in chats]

# Messages
@api_router.post("/messages", response_model=Message)
async def send_message(message: MessageCreate):
    chat = await db.chats.find_one({"id": message.chat_id}, {"_id": 0})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    new_message = Message(**message.model_dump())
    doc = new_message.model_dump()
    await db.messages.insert_one(doc)
    
    # Update chat last message
    await db.chats.update_one(
        {"id": message.chat_id},
        {"$set": {
            "last_message": message.text,
            "last_message_at": new_message.created_at
        }}
    )
    
    # Send via WebSocket
    recipient_id = chat["buyer_id"] if chat["seller_id"] == message.sender_id else chat["seller_id"]
    await manager.send_message(recipient_id, {"type": "new_message", "message": doc})
    
    return new_message

@api_router.get("/messages/{chat_id}", response_model=List[Message])
async def get_messages(chat_id: str, skip: int = 0, limit: int = 50):
    messages = await db.messages.find({"chat_id": chat_id}, {"_id": 0}).sort("created_at", 1).skip(skip).limit(limit).to_list(limit)
    return [Message(**msg) for msg in messages]

# Ratings
@api_router.post("/ratings", response_model=Rating)
async def create_rating(rating: RatingCreate):
    # Check if already rated
    existing = await db.ratings.find_one({
        "user_id": rating.user_id,
        "rated_by": rating.rated_by
    })
    if existing:
        raise HTTPException(status_code=400, detail="Already rated")
    
    new_rating = Rating(**rating.model_dump())
    doc = new_rating.model_dump()
    await db.ratings.insert_one(doc)
    
    # Update user rating
    pipeline = [
        {"$match": {"user_id": rating.user_id}},
        {"$group": {"_id": None, "avg": {"$avg": "$stars"}, "count": {"$sum": 1}}}
    ]
    result = await db.ratings.aggregate(pipeline).to_list(1)
    
    if result:
        await db.users.update_one(
            {"id": rating.user_id},
            {"$set": {"rating": result[0]["avg"], "ratings_count": result[0]["count"]}}
        )
    
    return new_rating

@api_router.get("/ratings/{user_id}", response_model=List[Rating])
async def get_ratings(user_id: str):
    ratings = await db.ratings.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return [Rating(**rating) for rating in ratings]

# Bans
@api_router.post("/bans")
async def ban_user(ban: BanCreate):
    admin = await db.users.find_one({"id": ban.banned_by}, {"_id": 0})
    if not admin or not admin.get("is_admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    new_ban = Ban(**ban.model_dump())
    doc = new_ban.model_dump()
    await db.bans.insert_one(doc)
    
    # Update user
    await db.users.update_one(
        {"id": ban.user_id},
        {"$set": {
            "is_banned": True,
            "banned_until": ban.until,
            "ban_reason": ban.reason
        }}
    )
    
    return new_ban

@api_router.delete("/bans/{user_id}")
async def unban_user(user_id: str, admin_id: str):
    admin = await db.users.find_one({"id": admin_id}, {"_id": 0})
    if not admin or not admin.get("is_admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db.users.update_one(
        {"id": user_id},
        {"$set": {"is_banned": False, "banned_until": None, "ban_reason": None}}
    )
    
    return {"status": "unbanned"}

# Admin stats
@api_router.get("/admin/stats")
async def get_admin_stats(admin_id: str):
    admin = await db.users.find_one({"id": admin_id}, {"_id": 0})
    if not admin or not admin.get("is_admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Total listings
    total_listings = await db.listings.count_documents({})
    
    # Today listings
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_listings = await db.listings.count_documents({
        "created_at": {"$gte": today_start.isoformat()}
    })
    
    # Week listings
    week_start = today_start - timedelta(days=7)
    week_listings = await db.listings.count_documents({
        "created_at": {"$gte": week_start.isoformat()}
    })
    
    # Month listings
    month_start = today_start - timedelta(days=30)
    month_listings = await db.listings.count_documents({
        "created_at": {"$gte": month_start.isoformat()}
    })
    
    # Total users
    total_users = await db.users.count_documents({})
    
    # Pending moderation
    pending = await db.listings.count_documents({"status": "pending"})
    
    return {
        "total_listings": total_listings,
        "today_listings": today_listings,
        "week_listings": week_listings,
        "month_listings": month_listings,
        "total_users": total_users,
        "pending_moderation": pending
    }

# WebSocket
@api_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(user_id)

# File serving
@api_router.get("/uploads/{filename}")
async def serve_file(filename: str):
    filepath = UPLOADS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

# Initialize default data
@api_router.post("/init")
async def initialize_data():
    # Check if already initialized
    existing = await db.categories.find_one({})
    if existing:
        return {"status": "already initialized"}
    
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
    
    return {"status": "initialized", "categories": len(categories)}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()