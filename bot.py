"""
Discord bot with DeepSeek LLM integration.
Supports direct commands, rewrite commands, and keyword triggers.
Uses v2 character card spec for base character/world context.
"""

import asyncio
import base64
import io
import json
import random
import re
import time
from datetime import date
import discord
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI
from PIL import Image
import boto3
from botocore.config import Config
import aiohttp

# =============================================================================
# CONFIGURATION
# =============================================================================

DISCORD_TOKEN = "YOUR_DISCORD_TOKEN_HERE"
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
CHARACTER_CARD_PATH = "your_character_card.json"
GACHA_CONFIG_PATH = "characters.json"
USER_DATA_PATH = "user_data.json"
DAILY_ROLLS = 3

# Rarity display config
RARITY_COLORS = {
    "N": 0x808080,    # gray
    "R": 0x3498db,    # blue
    "SR": 0x9b59b6,   # purple
    "SSR": 0xf1c40f,  # gold
    "UR": 0xe74c3c,   # red
}
RARITY_EMOJIS = {
    "N": "\u26aa",
    "R": "\U0001f535",
    "SR": "\U0001f7e3",
    "SSR": "\U0001f31f",
    "UR": "\U0001f48e",
}

# R2 Storage Configuration (for Chub.ai character imports)
R2_ACCOUNT_ID = "YOUR_R2_ACCOUNT_ID_HERE"
R2_ACCESS_KEY = "YOUR_R2_ACCESS_KEY_HERE"
R2_SECRET_KEY = "YOUR_R2_SECRET_KEY_HERE"
R2_BUCKET = "YOUR_R2_BUCKET_NAME_HERE"
R2_PUBLIC_URL = "YOUR_R2_PUBLIC_URL_HERE"  # e.g., https://pub-xxx.r2.dev

# Stats per rarity tier (for imported characters)
RARITY_STATS = {
    "N":   {"hp": 50,  "atk": 10, "def": 10},
    "R":   {"hp": 70,  "atk": 15, "def": 15},
    "SR":  {"hp": 90,  "atk": 20, "def": 20},
    "SSR": {"hp": 120, "atk": 28, "def": 25},
    "UR":  {"hp": 150, "atk": 35, "def": 30},
}

# Keywords/phrases that trigger automatic responses (case-insensitive)
TRIGGER_KEYWORDS = [
    "can someone explain",
    "what does this mean",
    "i don't understand",
    "help me understand",
    "confused about",
]

# Slop trigger keywords (common LLM-isms to rewrite)
SLOP_KEYWORDS = [
    # Original triggers
    "ozone",
    "hitch",
    "a beat",
    "slop",
    "delve",
    "stumbled upon",
    "couldn't help but",
    "a mix of",
    "sends shivers",
    "couldn't be further from the truth",
    "vibrant tapestry",
    "in this moment",
    "testament to",
    "it's worth noting",
    "dance of",
    "unbeknownst to",
    # Purple prose / LLM clichés
    "somewhere in the distance",
    "searing kiss",
    "all teeth and tongue",
    "breath hitches",
    "pleasure-pain",
    "ministrations",
    "primal",
    "chef's kiss",
    "really looked",
    "tantalizing sliver",
    "like a physical blow",
    "flicker of a smile",
    "just maybe",
    "voice cracking",
    "it's not a question",
    "it's a dare",
    "firm, insistent",
    "when it finally came",
    "animalistic",
    "predatory",
    "soft curves against hard planes",
    "nobody cares",
    "searches their eyes",
    "searches his eyes",
    "searches her eyes",
    "time stood still",
    "what seemed like an eternity",
    "a pang",
    "wicked grin",
    "she smirked",
    "he smirked",
    "leaned in",
    "conspiratorial whisper",
    "holds its breath",
    "low and",
    "draw blood",
    "angry red welts",
    "dark promise",
    "carotid",
    "like a moth to a flame",
    "they both knew",
    "all teeth",
    "recoiled as if struck",
    "bit their lip",
    "bit her lip",
    "bit his lip",
    "drops an octave",
    "darkened with",
    "let the threat hang",
    "hang in the air",
    "calloused hands",
    "globes of her",
    "silent scream",
    "her folds",
    "shivers down",
    "down the spine",
    "swell of her",
    "swell of his",
    "swell of their",
    "something more primal",
    "and something more",
]

# =============================================================================
# MODE-SPECIFIC INSTRUCTIONS (appended to character card context)
# =============================================================================

MODE_INSTRUCTION_ASK = """## Current Task
Lilith is responding to a direct question or request from the user in character. make it sloppy, long, and unecessarily vulgar. Skip any preamble."""

MODE_INSTRUCTION_REWRITE = """## Current Task
Lilith is being asked to rewrite/edit text according to the user's instructions. Stay in character while doing this task. Follow the rewrite instructions precisely and output ONLY the rewritten text without any preamble, explanation, or commentary. Skip any preamble."""

MODE_INSTRUCTION_INTERJECTION = """## Current Task
Lilith noticed something in the conversation you might be able to help with. You're jumping in unprompted, make sure to give a long explanation. Stay in character. Skip any preamble."""

MODE_INSTRUCTION_SLOP = """## Current Task
Lilith rewrites the user's message in her unique style. Make it sloppy, and dripping with her personality. Output ONLY the rewritten text without any preamble or explanation. Do not respond to the message, just rewrite it. Keep it no more than 2 times the length of the original message."""

MODE_INSTRUCTION_TALK = """## Current Task
You ARE this character chatting on Discord. Write like you're texting/DMing - casual, short responses (1-3 sentences usually). NO asterisks for actions, NO roleplay narration, NO describing what you're doing. Just talk like a real person would in a Discord server. Be authentic to your personality but keep it snappy and conversational."""

# =============================================================================
# CHARACTER CARD LOADING
# =============================================================================

def load_character_card(path: str) -> dict:
    """Load and parse a v2 character card JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_character_prompt(card: dict, include_scenario: bool = True) -> str:
    """Build a system prompt from v2 character card data."""
    data = card.get("data", {})

    parts = []

    # Character name and description
    if data.get("name"):
        parts.append(f"# Character: {data['name']}")

    if data.get("description"):
        parts.append(f"## Description\n{data['description']}")

    if data.get("personality"):
        parts.append(f"## Personality\n{data['personality']}")

    # Skip scenario for rewrite tasks to avoid roleplay context bleeding in
    if include_scenario and data.get("scenario"):
        parts.append(f"## Scenario\n{data['scenario']}")

    # Creator notes can have useful guidance
    if data.get("creator_notes"):
        parts.append(f"## Character Notes\n{data['creator_notes']}")

    # Depth prompt often has speech/behavior guidance
    extensions = data.get("extensions", {})
    depth_prompt = extensions.get("depth_prompt", {})
    if depth_prompt.get("prompt"):
        parts.append(depth_prompt["prompt"])

    # Character book / lorebook entries
    char_book = data.get("character_book", {})
    entries = char_book.get("entries", [])
    if entries:
        lore_parts = ["## World Lore"]
        for entry in entries:
            if entry.get("enabled", True) and entry.get("content"):
                lore_parts.append(entry["content"])
        if len(lore_parts) > 1:
            parts.append("\n".join(lore_parts))

    return "\n\n".join(parts)


# Load character card at startup
character_card = load_character_card(CHARACTER_CARD_PATH)
CHARACTER_BASE_PROMPT = build_character_prompt(character_card)
# Minimal version without scenario for rewrite tasks
CHARACTER_REWRITE_PROMPT = build_character_prompt(character_card, include_scenario=False)


def load_gacha_config(path: str) -> dict:
    """Load gacha characters config."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found. !roll command will be disabled.")
        return {"base_url": "", "characters": []}
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse {path}: {e}")
        return {"base_url": "", "characters": []}


# Load gacha config at startup
gacha_config = load_gacha_config(GACHA_CONFIG_PATH)


def save_gacha_config():
    """Save gacha config to file."""
    with open(GACHA_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(gacha_config, f, indent=2)


# =============================================================================
# CHUB.AI CHARACTER IMPORT FUNCTIONS
# =============================================================================

def get_r2_client():
    """Get boto3 client for R2 storage."""
    return boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )


def upload_char_image_sync(char_id: str, image_bytes: bytes) -> str:
    """Upload character image to R2, return public URL."""
    client = get_r2_client()
    key = f"{char_id}-img.webp"

    # Upload PNG as webp (Discord handles the conversion on display)
    client.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType='image/png'
    )

    return f"{R2_PUBLIC_URL}/{key}"


def parse_chub_url(url: str):
    """Extract username and character_id from chub.ai URL."""
    # Handles: https://chub.ai/characters/username/char-id
    match = re.match(r'https?://chub\.ai/characters/([^/]+)/([^/?]+)', url)
    if match:
        return match.group(1), match.group(2)
    return None


def extract_card_json(png_bytes: bytes):
    """Extract character card JSON from PNG metadata."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        if 'chara' in img.info:
            json_str = base64.b64decode(img.info['chara']).decode('utf-8')
            return json.loads(json_str)
    except Exception as e:
        print(f"Error extracting card JSON: {e}")
    return None


async def fetch_chub_character(username: str, char_id: str):
    """Fetch character data and image from Chub.ai."""
    png_url = f"https://avatars.charhub.io/avatars/{username}/{char_id}/chara_card_v2.png"

    async with aiohttp.ClientSession() as session:
        async with session.get(png_url) as resp:
            if resp.status != 200:
                return None, None
            png_data = await resp.read()

    # Extract JSON from PNG metadata
    card_data = extract_card_json(png_data)
    if not card_data:
        return None, None

    return card_data, png_data


def chub_to_bot_format(card_data: dict, chub_username: str, chub_id: str, rarity: str = "SR") -> dict:
    """Convert Chub.ai card to bot character format."""
    data = card_data.get("data", card_data)
    name = data.get("name", "Unknown")

    # Create ID from name (alphanumeric only)
    char_id = re.sub(r'[^a-zA-Z0-9]', '', name)

    # Use only description field (keep {{char}}/{{user}} for runtime replacement)
    description = data.get("description", "")

    return {
        "id": char_id,
        "name": name,
        "rarity": rarity,
        "stats": RARITY_STATS.get(rarity, RARITY_STATS["SR"]).copy(),
        "description": description.strip(),
        "chub_source": f"https://chub.ai/characters/{chub_username}/{chub_id}"
    }


# Lock to prevent race conditions in roll/claim operations
roll_lock = asyncio.Lock()

# =============================================================================
# USER DATA PERSISTENCE
# =============================================================================

def load_user_data() -> dict:
    """Load user data from JSON file."""
    try:
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return create_fresh_user_data()
    except json.JSONDecodeError:
        return create_fresh_user_data()


def save_user_data(data: dict):
    """Save user data to JSON file."""
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Pool copies per rarity (lower rarity = more common)
RARITY_POOL_COUNTS = {
    "N": 5,
    "R": 3,
    "SR": 2,
    "SSR": 1,
    "UR": 1,
}


def build_daily_pool() -> dict:
    """Build daily pool with rarity-based occurrence rates."""
    characters = gacha_config.get("characters", [])
    return {
        char["id"]: RARITY_POOL_COUNTS.get(char.get("rarity", "N"), 1)
        for char in characters
    }


def create_fresh_user_data() -> dict:
    """Create fresh user data with full daily pool."""
    return {
        "daily_reset": str(date.today()),
        "daily_pool": build_daily_pool(),
        "users": {}
    }


def check_daily_reset(data: dict) -> dict:
    """Check if daily reset is needed and reset pool/rolls if so."""
    today = str(date.today())
    if data.get("daily_reset") != today:
        # Reset daily pool with rarity-based counts
        data["daily_pool"] = build_daily_pool()
        data["daily_reset"] = today
        # Reset all users' daily rolls (but keep collections and bonus rolls)
        for user_id in data.get("users", {}):
            data["users"][user_id]["rolls_today"] = 0
        save_user_data(data)
    return data


def get_user(data: dict, user_id: str) -> dict:
    """Get or create user entry."""
    if user_id not in data.get("users", {}):
        if "users" not in data:
            data["users"] = {}
        data["users"][user_id] = {
            "rolls_today": 0,
            "bonus_rolls": 0,
            "collection": {},  # dict of char_id -> count
            "curses": {}  # active curses
        }
    else:
        # Migrate old list format to new dict format
        user = data["users"][user_id]
        if isinstance(user.get("collection"), list):
            old_collection = user["collection"]
            user["collection"] = {char_id: 1 for char_id in old_collection}
        # Ensure curses field exists
        if "curses" not in user:
            user["curses"] = {}
    return data["users"][user_id]


def get_available_characters(data: dict, user_id: str) -> list:
    """Get characters available in today's pool (dupes allowed)."""
    pool = data.get("daily_pool", {})

    available = []
    for char in gacha_config.get("characters", []):
        char_id = char["id"]
        if pool.get(char_id, 0) > 0:
            available.append(char)
    return available


def get_char_copies(data: dict, user_id: str, char_id: str) -> int:
    """Get number of copies a user has of a character."""
    user = get_user(data, user_id)
    collection = user.get("collection", {})
    if isinstance(collection, list):  # Handle old format
        return 1 if char_id in collection else 0
    return collection.get(char_id, 0)


def get_power_multiplier(copies: int) -> float:
    """Get power multiplier based on number of copies. Each dupe adds 10%."""
    if copies <= 1:
        return 1.0
    return 1.0 + (copies - 1) * 0.1  # +10% per dupe


def get_rolls_remaining(data: dict, user_id: str) -> int:
    """Get total rolls remaining (daily + bonus)."""
    user = get_user(data, user_id)
    daily_remaining = DAILY_ROLLS - user.get("rolls_today", 0)
    bonus = user.get("bonus_rolls", 0)
    return daily_remaining + bonus


def use_roll(data: dict, user_id: str):
    """Consume one roll (bonus first, then daily)."""
    user = get_user(data, user_id)
    if user.get("bonus_rolls", 0) > 0:
        user["bonus_rolls"] -= 1
    else:
        user["rolls_today"] = user.get("rolls_today", 0) + 1


def claim_character(data: dict, user_id: str, char_id: str) -> bool:
    """Add character to user's collection and remove from pool. Returns True if dupe."""
    user = get_user(data, user_id)
    collection = user.get("collection", {})

    # Check if this is a duplicate
    is_dupe = collection.get(char_id, 0) > 0

    # Increment count (or set to 1 if new)
    collection[char_id] = collection.get(char_id, 0) + 1
    user["collection"] = collection

    # Remove from daily pool
    if char_id in data.get("daily_pool", {}):
        data["daily_pool"][char_id] = max(0, data["daily_pool"][char_id] - 1)

    return is_dupe


def grant_bonus_roll(data: dict, user_id: str):
    """Grant a bonus roll to user."""
    user = get_user(data, user_id)
    user["bonus_rolls"] = user.get("bonus_rolls", 0) + 1


def get_char_by_id(char_id: str) -> dict | None:
    """Get character data by ID."""
    for char in gacha_config.get("characters", []):
        if char["id"] == char_id:
            return char
    return None


def get_char_description(char: dict) -> str:
    """Get character description, preferring alt_description if available."""
    return char.get("alt_description") or char.get("description", "A mysterious character.")


def find_char_by_name(name: str) -> dict | None:
    """Find a character by name (case-insensitive, supports partial match)."""
    name_lower = name.lower().strip()
    characters = gacha_config.get("characters", [])

    # Try exact match first
    for char in characters:
        if char["name"].lower() == name_lower or char["id"].lower() == name_lower:
            return char

    # Try partial match
    for char in characters:
        if name_lower in char["name"].lower() or name_lower in char["id"].lower():
            return char

    return None


def build_gacha_char_prompt(char: dict, user_name: str = "User") -> str:
    """Build a system prompt for a gacha character based on their description."""
    name = char.get("name", "Unknown")
    description = get_char_description(char)
    rarity = char.get("rarity", "N")

    # Replace {{char}} and {{user}} placeholders in description
    description = description.replace("{{char}}", name).replace("{{Char}}", name).replace("{{CHAR}}", name)
    description = description.replace("{{user}}", user_name).replace("{{User}}", user_name).replace("{{USER}}", user_name)

    return f"""# Character: {name}

## Who You Are
{description}

## Discord Chat Guidelines
- You ARE {name} chatting on Discord. Talk like yourself, not a narrator.
- Keep responses SHORT - 1-3 sentences max. This is texting, not a novel.
- NO asterisks (*action*), NO roleplay narration, NO "I smile warmly" type stuff.
- Just respond naturally like you would if someone DMed you.
- Use your unique speech patterns, slang, and personality quirks.
- You can use emoji sparingly if it fits your character."""


# =============================================================================
# CURSE SYSTEM
# =============================================================================

CURSE_TYPES = ["quirk_chungus", "lilith"]


def apply_curse(data: dict, target_id: str, curse_type: str, power: int = 100):
    """Apply a curse to a target user. Power scales the effect (base 100)."""
    target = get_user(data, target_id)
    curses = target.get("curses", {})

    # Scale factor: power/100, minimum 1x, no cap
    scale = max(1.0, power / 100)

    if curse_type == "quirk_chungus":
        # Add QuirkMans to the pool (base 10, scaled)
        quirks_added = int(10 * scale)
        pool = data.get("daily_pool", {})
        pool["QuirkMan"] = pool.get("QuirkMan", 0) + quirks_added
        data["daily_pool"] = pool
        # Force pulls to be QuirkMan (base 3, scaled)
        forced_pulls = int(3 * scale)
        curses["quirk_chungus"] = curses.get("quirk_chungus", 0) + forced_pulls

    elif curse_type == "lilith":
        # Slop rewrite triggers (base 5, scaled, max 25)
        triggers = min(25, int(5 * scale))
        curses["lilith"] = curses.get("lilith", 0) + triggers

    target["curses"] = curses
    return scale  # Return scale for display


def sacrifice_character(data: dict, user_id: str) -> tuple[str, int, int] | tuple[None, int, int]:
    """Sacrifice a random character from user's collection. Returns (char_name, powered_stats, copies) or (None, 0, 0)."""
    user = get_user(data, user_id)
    collection = user.get("collection", {})

    if not collection:
        return None, 0, 0

    # Pick random character to sacrifice
    char_id = random.choice(list(collection.keys()))
    char = get_char_by_id(char_id)
    char_name = char["name"] if char else char_id
    copies = collection.get(char_id, 1)

    # Calculate total stats with dupe power boost
    base_stats = char.get("stats", {"hp": 50, "atk": 10, "def": 10}) if char else {"hp": 50, "atk": 10, "def": 10}
    power_mult = get_power_multiplier(copies)
    powered_stats = int((base_stats["hp"] + base_stats["atk"] + base_stats["def"]) * power_mult)

    # Remove one copy
    collection[char_id] -= 1
    if collection[char_id] <= 0:
        del collection[char_id]

    return char_name, powered_stats, copies


def get_curse_name(curse_type: str) -> str:
    """Get display name for curse type."""
    names = {
        "quirk_chungus": "Curse of Quirk Chungus",
        "lilith": "Curse of Lilith"
    }
    return names.get(curse_type, curse_type)


def get_curse_description(curse_type: str) -> str:
    """Get description of what the curse does."""
    descriptions = {
        "quirk_chungus": "Adds 10 QuirkMans to the pool and forces your next 3 rolls to be QuirkMan!",
        "lilith": "20% chance Lilith rewrites your messages in her style (up to 5 times)!"
    }
    return descriptions.get(curse_type, "A mysterious curse...")


# =============================================================================
# BOT SETUP
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Webhook cache for character responses
class WebhookCache:
    def __init__(self):
        self.webhooks = {}  # {channel_id: webhook}

    async def get_or_create(self, channel):
        """Get cached webhook or create one for the channel."""
        channel_id = channel.id

        if channel_id in self.webhooks:
            return self.webhooks[channel_id]

        try:
            # Find existing bot webhook
            webhooks = await channel.webhooks()
            for wh in webhooks:
                if wh.name == "Lilith Characters":
                    self.webhooks[channel_id] = wh
                    return wh

            # Create new webhook
            webhook = await channel.create_webhook(name="Lilith Characters")
            self.webhooks[channel_id] = webhook
            return webhook
        except discord.Forbidden:
            return None  # No permission

webhook_cache = WebhookCache()

# Cooldown cache to prevent rapid character triggers (anti-loop)
# {channel_id: [list of trigger timestamps]}
character_cooldowns = {}
COOLDOWN_WINDOW_SECONDS = 10  # Time window to track
COOLDOWN_MAX_TRIGGERS = 6  # Max triggers in window before cooldown kicks in

# DeepSeek client (OpenAI-compatible API)
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def build_system_prompt(mode_instruction: str, for_rewrite: bool = False) -> str:
    """Combine character base prompt with mode-specific instructions.

    Use for_rewrite=True to skip scenario context (prevents roleplay bleeding into rewrites).
    """
    base = CHARACTER_REWRITE_PROMPT if for_rewrite else CHARACTER_BASE_PROMPT
    return f"{base}\n\n{mode_instruction}"


def get_character_name() -> str:
    """Get character name from loaded card."""
    return character_card.get("data", {}).get("name", "Character")


def replace_placeholders(text: str, user_name: str) -> str:
    """Replace {{char}} and {{user}} placeholders in text."""
    char_name = get_character_name()
    text = text.replace("{{char}}", char_name).replace("{{CHAR}}", char_name)
    text = text.replace("{{user}}", user_name).replace("{{USER}}", user_name)
    return text


async def get_llm_response(system_prompt: str, user_message: str) -> str:
    """Send a request to DeepSeek and return the response."""
    response = await deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=800,
        stream=False,
    )
    return response.choices[0].message.content


async def send_long_message(message: discord.Message, content: str):
    """Send a message, splitting it if it exceeds Discord's 2000 char limit."""
    if len(content) <= 2000:
        await message.reply(content)
        return

    # Split into chunks
    chunks = []
    while content:
        if len(content) <= 2000:
            chunks.append(content)
            break
        # Find a good split point
        split_at = content.rfind("\n", 0, 2000)
        if split_at == -1:
            split_at = content.rfind(" ", 0, 2000)
        if split_at == -1:
            split_at = 2000
        chunks.append(content[:split_at])
        content = content[split_at:].lstrip()

    for i, chunk in enumerate(chunks):
        if i == 0:
            await message.reply(chunk)
        else:
            await message.channel.send(chunk)


async def handle_ask(message: discord.Message, prompt: str):
    """Handle the !ask command. Includes replied-to message if present."""
    async with message.channel.typing():
        try:
            full_prompt = prompt
            # Include replied-to message content if this is a reply
            if message.reference and message.reference.message_id:
                try:
                    replied_msg = await message.channel.fetch_message(message.reference.message_id)
                    if replied_msg.content:
                        full_prompt = f"[Replying to message: \"{replied_msg.content}\"]\n\n{prompt}"
                except discord.NotFound:
                    pass

            system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK, for_rewrite=True)
            response = await get_llm_response(system_prompt, full_prompt)
            response = replace_placeholders(response, message.author.display_name)
            await send_long_message(message, response)
        except Exception as e:
            await message.reply(f"Sorry, I encountered an error: {e}")


async def handle_askcontext(message: discord.Message, prompt: str, num_messages: int = 5):
    """Handle the !askcontext command. Includes last N messages as context."""
    async with message.channel.typing():
        try:
            # Fetch recent messages (excluding the command message itself)
            messages = []
            async for msg in message.channel.history(limit=num_messages + 1, before=message):
                if msg.content and not msg.author.bot:
                    messages.append(f"{msg.author.display_name}: {msg.content}")
            messages.reverse()

            context_str = "\n".join(messages) if messages else "(no recent messages)"
            full_prompt = f"[Recent conversation context:]\n{context_str}\n\n[User's question:] {prompt}"

            system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK, for_rewrite=True)
            response = await get_llm_response(system_prompt, full_prompt)
            response = replace_placeholders(response, message.author.display_name)
            await send_long_message(message, response)
        except Exception as e:
            await message.reply(f"Sorry, I encountered an error: {e}")


async def handle_rewrite(message: discord.Message, instructions: str):
    """Handle the !rewrite command (must be a reply to another message)."""
    # Check if this is a reply to another message
    if message.reference is None or message.reference.message_id is None:
        await message.reply(
            "Please use `!rewrite` as a reply to the message you want to rewrite."
        )
        return

    async with message.channel.typing():
        try:
            # Fetch the original message
            original_message = await message.channel.fetch_message(
                message.reference.message_id
            )
            original_content = original_message.content

            if not original_content:
                await message.reply(
                    "The referenced message has no text content to rewrite."
                )
                return

            # Build the prompt with both original text and instructions
            user_prompt = f"""Original text:
{original_content}

Rewrite instructions:
{instructions}"""

            system_prompt = build_system_prompt(MODE_INSTRUCTION_REWRITE, for_rewrite=True)
            response = await get_llm_response(system_prompt, user_prompt)
            response = replace_placeholders(response, message.author.display_name)
            await send_long_message(message, response)
        except discord.NotFound:
            await message.reply("Could not find the referenced message.")
        except Exception as e:
            await message.reply(f"Sorry, I encountered an error: {e}")


async def handle_keyword_trigger(message: discord.Message):
    """Handle passive keyword trigger responses."""
    async with message.channel.typing():
        try:
            system_prompt = build_system_prompt(MODE_INSTRUCTION_INTERJECTION)
            response = await get_llm_response(system_prompt, message.content)
            response = replace_placeholders(response, message.author.display_name)
            await send_long_message(message, response)
        except Exception as e:
            # Silently fail for keyword triggers to avoid spam on errors
            print(f"Keyword trigger error: {e}")


async def handle_slop_trigger(message: discord.Message, curse_triggered: bool = False):
    """Handle slop trigger - rewrite user's message in character's sloppy style."""
    async with message.channel.typing():
        try:
            system_prompt = build_system_prompt(MODE_INSTRUCTION_SLOP, for_rewrite=True)
            response = await get_llm_response(system_prompt, message.content)
            response = replace_placeholders(response, message.author.display_name)
            if curse_triggered:
                response = f"*🔮 Curse of Lilith activates!*\n\n{response}"
            await send_long_message(message, response)
        except Exception as e:
            print(f"Slop trigger error: {e}")


async def handle_name_trigger(message: discord.Message, char: dict, full_message: str = ""):
    """Handle when someone types a character's name - character responds via webhook."""
    base_url = gacha_config.get("base_url", "")
    char_id = char.get("id", "")
    char_name = char.get("name", "Unknown")
    avatar_url = f"{base_url}/{char_id}-img.webp" if base_url and char_id else None

    # Get or create webhook for this channel
    webhook = await webhook_cache.get_or_create(message.channel)

    # Fallback header if no webhook
    rarity = char.get("rarity", "N")
    emoji = RARITY_EMOJIS.get(rarity, "")

    try:
        # Fetch last 30 messages of conversation context (include all users and bots)
        context_messages = []
        seen_characters = set()  # Track which characters we've seen for descriptions

        async for msg in message.channel.history(limit=30, before=message):
            if msg.content:
                author_name = msg.author.display_name
                context_messages.append(f"{author_name}: {msg.content}")

                # Check if this is a character (webhook message or matching name)
                if msg.webhook_id or find_char_by_name(author_name):
                    seen_characters.add(author_name)
        context_messages.reverse()

        # Build character descriptions for context
        char_descriptions = []
        for seen_name in seen_characters:
            found_char = find_char_by_name(seen_name)
            if found_char and found_char["id"] != char["id"]:  # Don't describe self
                desc = found_char.get("description", "")
                if desc:
                    char_descriptions.append(f"- {found_char['name']}: {desc}")

        # Build the full prompt with context - use the entire message they sent
        context_str = "\n".join(context_messages) if context_messages else ""
        char_info_str = "\n".join(char_descriptions) if char_descriptions else ""

        if context_str:
            if char_info_str:
                full_prompt = f"[Other characters in this conversation:]\n{char_info_str}\n\n[Recent chat in this Discord channel:]\n{context_str}\n\n[{message.author.display_name} says:] {full_message}"
            else:
                full_prompt = f"[Recent chat in this Discord channel:]\n{context_str}\n\n[{message.author.display_name} says:] {full_message}"
        else:
            full_prompt = f"[{message.author.display_name} says:] {full_message}"

        # Build character-specific system prompt
        char_prompt = build_gacha_char_prompt(char, message.author.display_name)
        system_prompt = f"{char_prompt}\n\n{MODE_INSTRUCTION_TALK}"

        response = await get_llm_response(system_prompt, full_prompt)
        response = response.replace("{{user}}", message.author.display_name).replace("{{USER}}", message.author.display_name)

        # Send via webhook if available, otherwise fallback to regular message
        if webhook:
            if len(response) <= 2000:
                await webhook.send(content=response, username=char_name, avatar_url=avatar_url)
            else:
                # Split long responses
                await webhook.send(content=response[:2000], username=char_name, avatar_url=avatar_url)
                remaining = response[2000:]
                while remaining:
                    chunk = remaining[:2000]
                    remaining = remaining[2000:]
                    await webhook.send(content=chunk, username=char_name, avatar_url=avatar_url)
        else:
            # Fallback to regular message with header
            header = f"**{emoji} {char_name}:**\n"
            await send_long_message(message, header + response)
    except Exception as e:
        await message.reply(f"**{emoji} {char_name}:** Error: {e}")


async def handle_talk(message: discord.Message, char_name: str, prompt: str):
    """Handle the !talk command - talk to a character in your collection."""
    user_id = str(message.author.id)
    data = load_user_data()
    data = check_daily_reset(data)
    user = get_user(data, user_id)
    collection = user.get("collection", {})

    # Handle old list format
    if isinstance(collection, list):
        collection = {cid: 1 for cid in collection}

    # Find the character
    char = find_char_by_name(char_name)

    if not char:
        # List available characters
        available = [get_char_by_id(cid)["name"] for cid in collection if get_char_by_id(cid)]
        if available:
            await message.reply(f"Character '{char_name}' not found. Your collection: {', '.join(available)}")
        else:
            await message.reply(f"Character '{char_name}' not found and you have no characters! Use `!roll` to collect some.")
        return

    # Check if user owns this character (or allow Lilith as default)
    char_id = char["id"]
    if char_id not in collection and char_id != "Lilith":
        available = [get_char_by_id(cid)["name"] for cid in collection if get_char_by_id(cid)]
        if available:
            await message.reply(f"You don't own **{char['name']}**! Your collection: {', '.join(available)}")
        else:
            await message.reply(f"You don't own **{char['name']}**! Use `!roll` to collect characters.")
        return

    # Get webhook and character info
    base_url = gacha_config.get("base_url", "")
    char_id = char["id"]
    avatar_url = f"{base_url}/{char_id}-img.webp" if base_url and char_id else None
    webhook = await webhook_cache.get_or_create(message.channel)

    try:
        # Fetch recent conversation context (include bot messages)
        context_messages = []
        async for msg in message.channel.history(limit=10, before=message):
            if msg.content:
                author_name = msg.author.display_name
                context_messages.append(f"{author_name}: {msg.content}")
        context_messages.reverse()

        # Build the full prompt with context
        context_str = "\n".join(context_messages) if context_messages else ""
        if context_str:
            full_prompt = f"[Recent chat in this Discord channel:]\n{context_str}\n\n[{message.author.display_name} says to you:] {prompt}"
        else:
            full_prompt = f"[{message.author.display_name} says to you:] {prompt}"

        # Build character-specific system prompt
        char_prompt = build_gacha_char_prompt(char, message.author.display_name)
        system_prompt = f"{char_prompt}\n\n{MODE_INSTRUCTION_TALK}"

        response = await get_llm_response(system_prompt, full_prompt)
        response = response.replace("{{user}}", message.author.display_name).replace("{{USER}}", message.author.display_name)

        # Send via webhook (no header)
        if webhook:
            if len(response) <= 2000:
                await webhook.send(content=response, username=char["name"], avatar_url=avatar_url)
            else:
                await webhook.send(content=response[:2000], username=char["name"], avatar_url=avatar_url)
                remaining = response[2000:]
                while remaining:
                    chunk = remaining[:2000]
                    remaining = remaining[2000:]
                    await webhook.send(content=chunk, username=char["name"], avatar_url=avatar_url)
        else:
            # Fallback if no webhook permission
            rarity = char.get("rarity", "N")
            emoji = RARITY_EMOJIS.get(rarity, "")
            header = f"**{emoji} {char['name']}:**\n"
            await send_long_message(message, header + response)
    except Exception as e:
        await message.reply(f"Error talking to {char['name']}: {e}")


async def handle_roll(message: discord.Message):
    """Handle the !roll gacha command with collection system."""
    base_url = gacha_config.get("base_url", "")
    user_id = str(message.author.id)

    # Use lock to prevent race conditions
    async with roll_lock:
        # Load and check for daily reset
        data = load_user_data()
        data = check_daily_reset(data)

        # Check rolls remaining
        rolls_left = get_rolls_remaining(data, user_id)
        if rolls_left <= 0:
            await message.reply("You've used all your rolls today! Win a battle for bonus rolls.")
            return

        # Get available characters
        available = get_available_characters(data, user_id)
        if not available:
            await message.reply("No characters left in today's pool! Come back tomorrow.")
            return

        # Check for Quirk Chungus curse - forces QuirkMan roll
        user = get_user(data, user_id)
        quirk_curse = user.get("curses", {}).get("quirk_chungus", 0)
        curse_forced = False

        if quirk_curse > 0:
            # Force QuirkMan if available in pool
            quirk_char = get_char_by_id("QuirkMan")
            if quirk_char and data.get("daily_pool", {}).get("QuirkMan", 0) > 0:
                char = quirk_char
                curse_forced = True
                # Decrement curse counter
                user["curses"]["quirk_chungus"] -= 1
                if user["curses"]["quirk_chungus"] <= 0:
                    del user["curses"]["quirk_chungus"]
            else:
                # QuirkMan not in pool, pick random
                char = random.choice(available)
        else:
            # Pick random from available
            char = random.choice(available)

        char_id = char.get("id", "unknown")
        char_name = char.get("name", "Unknown")
        rarity = char.get("rarity", "N")
        description = char.get("description", "A mysterious character.")

        # Consume roll and claim character
        use_roll(data, user_id)
        is_dupe = claim_character(data, user_id, char_id)
        save_user_data(data)

        # Get updated counts
        user = get_user(data, user_id)
        collection = user.get("collection", {})
        copies = collection.get(char_id, 1)
        power_mult = get_power_multiplier(copies)
        unique_count = len(collection)
        total_count = len(gacha_config.get("characters", []))
        rolls_after = get_rolls_remaining(data, user_id)

    # Release lock before slow API call
    async with message.channel.typing():
        try:
            # Build image URL
            image_url = f"{base_url}/{char_id}-img.webp"

            # Generate intro via DeepSeek (narrated by base character)
            prompt = f"""{message.author.display_name} just rolled a gacha and summoned this character:
Character: {char_name}
Rarity: {rarity}
Personality: {description}

Introduce this character to the user in 4-5 sentences. Be dramatic and playful. Address the user by name. Don't repeat the bio verbatim—tease the character's personality in your own words."""

            system_prompt = build_system_prompt("")
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            intro_text = response.choices[0].message.content
            intro_text = replace_placeholders(intro_text, message.author.display_name)

            # Build embed
            emoji = RARITY_EMOJIS.get(rarity, "\u2753")
            color = RARITY_COLORS.get(rarity, 0x808080)

            embed = discord.Embed(
                title=f"{emoji} {rarity} — {char_name}",
                description=intro_text,
                color=color,
            )
            embed.set_image(url=image_url)

            # Footer shows power up info or NEW (and curse if forced)
            if curse_forced:
                footer_prefix = "🔮 CURSED ROLL!"
            elif is_dupe:
                footer_prefix = f"⬆️ POWER UP! x{copies} ({int(power_mult * 100)}% power)"
            else:
                footer_prefix = "✨ NEW!"
            embed.set_footer(text=f"{footer_prefix} • Collection: {unique_count}/{total_count} • Rolls left: {rolls_after}")

            await message.reply(embed=embed)

        except Exception as e:
            await message.reply(f"Something went wrong: {e}")


async def handle_collection(message: discord.Message):
    """Handle the !collection command."""
    user_id = str(message.author.id)
    data = load_user_data()
    data = check_daily_reset(data)
    user = get_user(data, user_id)

    collection = user.get("collection", {})
    # Handle old list format
    if isinstance(collection, list):
        collection = {cid: 1 for cid in collection}

    total_chars = len(gacha_config.get("characters", []))

    if not collection:
        await message.reply("You haven't collected any characters yet! Use `!roll` to start.")
        return

    # Group by rarity
    rarity_order = ["UR", "SSR", "SR", "R", "N"]
    grouped = {r: [] for r in rarity_order}

    for char_id, copies in collection.items():
        char = get_char_by_id(char_id)
        if char:
            rarity = char.get("rarity", "N")
            grouped[rarity].append((char, copies))

    # Build embed
    unique_count = len(collection)
    total_copies = sum(collection.values())
    embed = discord.Embed(
        title=f"{message.author.display_name}'s Collection",
        description=f"**{unique_count}/{total_chars}** unique • **{total_copies}** total copies",
        color=0x9b59b6,
    )

    for rarity in rarity_order:
        chars = grouped[rarity]
        if chars:
            emoji = RARITY_EMOJIS.get(rarity, "")
            # Show copy count if > 1
            names = ", ".join(
                f"{c['name']} x{copies}" if copies > 1 else c["name"]
                for c, copies in chars
            )
            embed.add_field(name=f"{emoji} {rarity}", value=names, inline=False)

    rolls_left = get_rolls_remaining(data, user_id)
    embed.set_footer(text=f"Rolls remaining today: {rolls_left}")

    await message.reply(embed=embed)


async def handle_pool(message: discord.Message):
    """Handle the !pool command."""
    user_id = str(message.author.id)
    data = load_user_data()
    data = check_daily_reset(data)

    pool = data.get("daily_pool", {})
    user = get_user(data, user_id)
    owned = set(user.get("collection", []))

    # Get available (in pool, quantity > 0)
    available_global = []
    available_for_user = []

    for char in gacha_config.get("characters", []):
        char_id = char["id"]
        if pool.get(char_id, 0) > 0:
            available_global.append(char)
            if char_id not in owned:
                available_for_user.append(char)

    # Build embed
    embed = discord.Embed(
        title="Today's Character Pool",
        color=0x3498db,
    )

    if available_global:
        # Group by rarity
        rarity_order = ["UR", "SSR", "SR", "R", "N"]
        grouped = {r: [] for r in rarity_order}
        for char in available_global:
            rarity = char.get("rarity", "N")
            # Mark if user already owns
            name = char["name"]
            if char["id"] in owned:
                name = f"~~{name}~~"
            grouped[rarity].append(name)

        for rarity in rarity_order:
            names = grouped[rarity]
            if names:
                emoji = RARITY_EMOJIS.get(rarity, "")
                embed.add_field(name=f"{emoji} {rarity}", value=", ".join(names), inline=False)

        embed.description = f"**{len(available_global)}** characters in pool • **{len(available_for_user)}** available for you\n*(Strikethrough = already owned)*"
    else:
        embed.description = "The pool is empty! All characters have been claimed today."

    rolls_left = get_rolls_remaining(data, user_id)
    embed.set_footer(text=f"Your rolls remaining: {rolls_left}")

    await message.reply(embed=embed)


async def handle_battle(message: discord.Message, opponent: discord.Member):
    """Handle the !battle command."""
    user_id = str(message.author.id)
    opponent_id = str(opponent.id)

    # Validation
    if message.author.id == opponent.id:
        await message.reply("You can't battle yourself!")
        return

    # Check if challenging the bot (Lilith)
    is_bot_battle = opponent.id == bot.user.id

    data = load_user_data()
    data = check_daily_reset(data)

    user1 = get_user(data, user_id)
    collection1 = user1.get("collection", {})

    # Handle old list format
    if isinstance(collection1, list):
        collection1 = {cid: 1 for cid in collection1}

    if not collection1:
        await message.reply("You don't have any characters! Use `!roll` first.")
        return

    if not is_bot_battle:
        user2 = get_user(data, opponent_id)
        collection2 = user2.get("collection", {})
        if isinstance(collection2, list):
            collection2 = {cid: 1 for cid in collection2}
        if not collection2:
            await message.reply(f"{opponent.display_name} doesn't have any characters yet!")
            return
    else:
        # Bot always uses Lilith at max power (x5)
        collection2 = {"Lilith": 5}

    async with message.channel.typing():
        try:
            # Random character from each collection
            char1_id = random.choice(list(collection1.keys()))
            char2_id = random.choice(list(collection2.keys()))
            char1 = get_char_by_id(char1_id)
            char2 = get_char_by_id(char2_id)

            if not char1 or not char2:
                await message.reply("Error loading character data.")
                return

            # Get base stats
            base_stats1 = char1.get("stats", {"hp": 50, "atk": 10, "def": 10})
            base_stats2 = char2.get("stats", {"hp": 50, "atk": 10, "def": 10})

            # Apply dupe power multiplier to stats
            copies1 = collection1.get(char1_id, 1)
            copies2 = collection2.get(char2_id, 1)
            power1 = get_power_multiplier(copies1)
            power2 = get_power_multiplier(copies2)

            stats1 = {
                "hp": int(base_stats1["hp"] * power1),
                "atk": int(base_stats1["atk"] * power1),
                "def": int(base_stats1["def"] * power1)
            }
            stats2 = {
                "hp": int(base_stats2["hp"] * power2),
                "atk": int(base_stats2["atk"] * power2),
                "def": int(base_stats2["def"] * power2)
            }

            # Random power modifiers (0.8 - 1.2)
            mod1 = round(random.uniform(0.8, 1.2), 2)
            mod2 = round(random.uniform(0.8, 1.2), 2)

            # Build battle prompt
            prompt = f"""You are a dramatic battle narrator. Two characters are fighting!

**{char1["name"]}** (fighting for {message.author.display_name})
- Stats: {stats1["hp"]} HP, {stats1["atk"]} ATK, {stats1["def"]} DEF
- Power Roll: {mod1}x multiplier
- Who they are: {char1.get("description", "Unknown")}

**{char2["name"]}** (fighting for {opponent.display_name})
- Stats: {stats2["hp"]} HP, {stats2["atk"]} ATK, {stats2["def"]} DEF
- Power Roll: {mod2}x multiplier
- Who they are: {char2.get("description", "Unknown")}

Write an exciting, dramatic battle scene (6-8 sentences). Describe the fight viscerally - their movements, attacks, reactions. Use their personalities to inform HOW they fight. Reference the power rolls affecting their performance. Build tension, then deliver a decisive conclusion.

The winner should be determined by: (ATK × Power Roll) vs opponent's DEF, with HP as tiebreaker. Higher effective damage wins.

End your narration with the victor standing triumphant. Then on the final line, write exactly:
WINNER: [owner's display name, either {message.author.display_name} or {opponent.display_name}]"""

            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            battle_text = response.choices[0].message.content

            # Parse winner from output
            winner_name = None
            winner_id = None
            lines = battle_text.strip().split("\n")
            for line in reversed(lines):
                if line.strip().upper().startswith("WINNER:"):
                    winner_name = line.split(":", 1)[1].strip()
                    break

            # Determine winner ID
            if winner_name:
                if winner_name.lower() == message.author.display_name.lower():
                    winner_id = user_id
                    winner_display = message.author.display_name
                    winner_char = char1
                elif winner_name.lower() == opponent.display_name.lower():
                    winner_id = opponent_id
                    winner_display = opponent.display_name
                    winner_char = char2
                else:
                    # Fallback: calculate based on stats
                    score1 = (stats1["atk"] * mod1) - (stats2["def"] * 0.5) + (stats1["hp"] / 20)
                    score2 = (stats2["atk"] * mod2) - (stats1["def"] * 0.5) + (stats2["hp"] / 20)
                    if score1 >= score2:
                        winner_id = user_id
                        winner_display = message.author.display_name
                        winner_char = char1
                    else:
                        winner_id = opponent_id
                        winner_display = opponent.display_name
                        winner_char = char2
            else:
                # No winner line found, calculate
                score1 = (stats1["atk"] * mod1) - (stats2["def"] * 0.5) + (stats1["hp"] / 20)
                score2 = (stats2["atk"] * mod2) - (stats1["def"] * 0.5) + (stats2["hp"] / 20)
                if score1 >= score2:
                    winner_id = user_id
                    winner_display = message.author.display_name
                    winner_char = char1
                else:
                    winner_id = opponent_id
                    winner_display = opponent.display_name
                    winner_char = char2

            # Grant bonus roll to winner (only if not the bot)
            if not is_bot_battle or winner_id == user_id:
                grant_bonus_roll(data, winner_id)
                save_user_data(data)
                footer_text = f"🏆 Winner: {winner_display} (+1 bonus roll)"
            else:
                # Bot won, no bonus roll
                footer_text = f"🏆 Winner: {winner_display} (Lilith claims victory!)"

            # Remove WINNER line from display text
            display_text = "\n".join(
                line for line in battle_text.split("\n")
                if not line.strip().upper().startswith("WINNER:")
            ).strip()

            # Build embed
            embed = discord.Embed(
                title=f"⚔️ {char1['name']} vs {char2['name']}",
                description=display_text,
                color=RARITY_COLORS.get(winner_char.get("rarity", "N"), 0x808080),
            )
            # Show dupe level if > 1
            dupe_str1 = f" (x{copies1})" if copies1 > 1 else ""
            dupe_str2 = f" (x{copies2})" if copies2 > 1 else ""

            embed.add_field(
                name=f"{message.author.display_name}'s Fighter",
                value=f"**{char1['name']}**{dupe_str1}\nATK {stats1['atk']} • DEF {stats1['def']} • HP {stats1['hp']}\nRoll: {mod1}x",
                inline=True
            )
            embed.add_field(
                name=f"{opponent.display_name}'s Fighter",
                value=f"**{char2['name']}**{dupe_str2}\nATK {stats2['atk']} • DEF {stats2['def']} • HP {stats2['hp']}\nRoll: {mod2}x",
                inline=True
            )
            embed.set_footer(text=footer_text)

            await message.reply(embed=embed)

        except Exception as e:
            await message.reply(f"Battle error: {e}")


async def handle_harem(message: discord.Message):
    """Handle the !harem command - collected characters argue about being the favorite."""
    user_id = str(message.author.id)
    data = load_user_data()
    data = check_daily_reset(data)
    user = get_user(data, user_id)

    collection = user.get("collection", {})
    # Handle old list format
    if isinstance(collection, list):
        collection = {cid: 1 for cid in collection}

    if len(collection) < 2:
        await message.reply("You need at least 2 characters in your collection for a harem argument! Use `!roll` to collect more.")
        return

    async with message.channel.typing():
        try:
            # Build character list with descriptions
            char_list = []
            for char_id in collection.keys():
                char = get_char_by_id(char_id)
                if char:
                    desc = get_char_description(char)
                    char_list.append(f"**{char['name']}**: {desc}")

            chars_text = "\n\n".join(char_list)

            prompt = f"""You are writing a comedic scene. These characters are all in {message.author.display_name}'s collection and are arguing about who should be their favorite:

{chars_text}

Write a funny, chaotic argument scene (8-10 sentences) where they each make their case for why THEY should be the favorite and bicker with each other. Stay true to each character's personality and speech patterns. Make it dramatic, petty, and entertaining. They should interrupt each other, throw shade, and get increasingly unhinged."""

            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            scene_text = response.choices[0].message.content

            # Build embed
            embed = discord.Embed(
                title=f"💕 {message.author.display_name}'s Harem Argument",
                description=scene_text,
                color=0xff69b4,  # Hot pink
            )
            char_names = [get_char_by_id(cid)["name"] for cid in collection if get_char_by_id(cid)]
            embed.set_footer(text=f"Featuring: {', '.join(char_names)}")

            await message.reply(embed=embed)

        except Exception as e:
            await message.reply(f"Harem error: {e}")


async def handle_curse(message: discord.Message, target: discord.Member):
    """Handle the !curse command - sacrifice a character to curse someone."""
    user_id = str(message.author.id)
    target_id = str(target.id)

    # Validation
    if message.author.id == target.id:
        await message.reply("You can't curse yourself! ...or can you? No, you can't.")
        return

    if target.bot and target.id != bot.user.id:
        await message.reply("You can only curse humans... or Lilith.")
        return

    data = load_user_data()
    data = check_daily_reset(data)

    user = get_user(data, user_id)
    collection = user.get("collection", {})

    if not collection:
        await message.reply("You need at least one character to sacrifice for a curse! Use `!roll` first.")
        return

    # Sacrifice a random character
    sacrificed_name, sacrifice_power, copies = sacrifice_character(data, user_id)

    # Pick random curse and apply with power scaling
    curse_type = random.choice(CURSE_TYPES)
    scale = apply_curse(data, target_id, curse_type, sacrifice_power)

    save_user_data(data)

    # Build response embed
    curse_name = get_curse_name(curse_type)
    curse_desc = get_curse_description(curse_type)
    dupe_str = f" x{copies}" if copies > 1 else ""

    embed = discord.Embed(
        title=f"🔮 {curse_name}",
        description=f"**{message.author.display_name}** sacrificed **{sacrificed_name}**{dupe_str} (power: {sacrifice_power}) to curse **{target.display_name}**!",
        color=0x800080,  # Purple
    )
    embed.add_field(name="Effect", value=f"{curse_desc}\n*Scaled to {scale:.1f}x power!*", inline=False)

    # Show remaining curses on target
    target_user = get_user(data, target_id)
    active_curses = target_user.get("curses", {})
    if active_curses:
        curse_list = []
        for c_type, c_value in active_curses.items():
            curse_list.append(f"• {get_curse_name(c_type)}: {c_value} remaining")
        embed.add_field(name=f"{target.display_name}'s Active Curses", value="\n".join(curse_list), inline=False)

    await message.reply(embed=embed)


def contains_trigger_keyword(content: str) -> bool:
    """Check if the message contains any trigger keywords."""
    content_lower = content.lower()
    return any(keyword.lower() in content_lower for keyword in TRIGGER_KEYWORDS)


def contains_slop_keyword(content: str) -> bool:
    """Check if the message contains any slop trigger keywords."""
    content_lower = content.lower()
    return any(keyword.lower() in content_lower for keyword in SLOP_KEYWORDS)


@bot.event
async def on_ready():
    char_name = character_card.get("data", {}).get("name", "Unknown")
    gacha_count = len(gacha_config.get("characters", []))
    print(f"Bot is ready! Logged in as {bot.user}")
    print(f"Loaded character: {char_name}")
    print(f"Gacha characters loaded: {gacha_count}")
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash command(s)")
    except Exception as e:
        print(f"Failed to sync slash commands: {e}")


# =============================================================================
# SLASH COMMANDS
# =============================================================================

@bot.tree.command(name="ask", description="Ask the character a question")
@app_commands.describe(prompt="Your question or request")
async def slash_ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    try:
        system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK, for_rewrite=True)
        response = await get_llm_response(system_prompt, prompt)
        response = replace_placeholders(response, interaction.user.display_name)
        if len(response) <= 2000:
            await interaction.followup.send(response)
        else:
            chunks = []
            content = response
            while content:
                if len(content) <= 2000:
                    chunks.append(content)
                    break
                split_at = content.rfind("\n", 0, 2000)
                if split_at == -1:
                    split_at = content.rfind(" ", 0, 2000)
                if split_at == -1:
                    split_at = 2000
                chunks.append(content[:split_at])
                content = content[split_at:].lstrip()
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.followup.send(chunk)
                else:
                    await interaction.channel.send(chunk)
    except Exception as e:
        await interaction.followup.send(f"Sorry, I encountered an error: {e}")


@bot.tree.command(name="askcontext", description="Ask with recent chat history as context")
@app_commands.describe(prompt="Your question", messages="Number of messages for context (default 5, max 50)")
async def slash_askcontext(interaction: discord.Interaction, prompt: str, messages: int = 5):
    await interaction.response.defer()
    try:
        num_messages = min(max(messages, 1), 50)
        # Fetch recent messages
        msg_list = []
        async for msg in interaction.channel.history(limit=num_messages + 1):
            if msg.content and not msg.author.bot:
                msg_list.append(f"{msg.author.display_name}: {msg.content}")
        msg_list.reverse()

        context_str = "\n".join(msg_list) if msg_list else "(no recent messages)"
        full_prompt = f"[Recent conversation context:]\n{context_str}\n\n[User's question:] {prompt}"

        system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK, for_rewrite=True)
        response = await get_llm_response(system_prompt, full_prompt)
        response = replace_placeholders(response, interaction.user.display_name)
        if len(response) <= 2000:
            await interaction.followup.send(response)
        else:
            chunks = []
            content = response
            while content:
                if len(content) <= 2000:
                    chunks.append(content)
                    break
                split_at = content.rfind("\n", 0, 2000)
                if split_at == -1:
                    split_at = content.rfind(" ", 0, 2000)
                if split_at == -1:
                    split_at = 2000
                chunks.append(content[:split_at])
                content = content[split_at:].lstrip()
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.followup.send(chunk)
                else:
                    await interaction.channel.send(chunk)
    except Exception as e:
        await interaction.followup.send(f"Sorry, I encountered an error: {e}")


@bot.tree.command(name="rewrite", description="Rewrite text in character's style")
@app_commands.describe(text="The text to rewrite", instructions="How to rewrite it")
async def slash_rewrite(interaction: discord.Interaction, text: str, instructions: str):
    await interaction.response.defer()
    try:
        user_prompt = f"""Original text:
{text}

Rewrite instructions:
{instructions}"""
        system_prompt = build_system_prompt(MODE_INSTRUCTION_REWRITE, for_rewrite=True)
        response = await get_llm_response(system_prompt, user_prompt)
        response = replace_placeholders(response, interaction.user.display_name)
        if len(response) <= 2000:
            await interaction.followup.send(response)
        else:
            await interaction.followup.send(response[:2000])
    except Exception as e:
        await interaction.followup.send(f"Sorry, I encountered an error: {e}")


@bot.tree.context_menu(name="Ask about this")
async def context_ask(interaction: discord.Interaction, message: discord.Message):
    """Context menu command - right-click a message to ask about it."""
    if not message.content:
        await interaction.response.send_message("That message has no text content.", ephemeral=True)
        return

    await interaction.response.defer()
    try:
        prompt = f"[User is asking about this message: \"{message.content}\"]\n\nWhat's going on here? Explain or comment on this."
        system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK, for_rewrite=True)
        response = await get_llm_response(system_prompt, prompt)
        response = replace_placeholders(response, interaction.user.display_name)
        if len(response) <= 2000:
            await interaction.followup.send(response)
        else:
            await interaction.followup.send(response[:2000])
    except Exception as e:
        await interaction.followup.send(f"Sorry, I encountered an error: {e}")


@bot.tree.context_menu(name="Rewrite this")
async def context_rewrite(interaction: discord.Interaction, message: discord.Message):
    """Context menu command - right-click a message to rewrite it."""
    if not message.content:
        await interaction.response.send_message("That message has no text content.", ephemeral=True)
        return

    await interaction.response.defer()
    try:
        user_prompt = f"""Original text:
{message.content}

Rewrite instructions:
Rewrite this in your own style."""
        system_prompt = build_system_prompt(MODE_INSTRUCTION_REWRITE, for_rewrite=True)
        response = await get_llm_response(system_prompt, user_prompt)
        response = replace_placeholders(response, interaction.user.display_name)
        if len(response) <= 2000:
            await interaction.followup.send(response)
        else:
            await interaction.followup.send(response[:2000])
    except Exception as e:
        await interaction.followup.send(f"Sorry, I encountered an error: {e}")


@bot.tree.command(name="roll", description="Roll the gacha for a random character")
async def slash_roll(interaction: discord.Interaction):
    base_url = gacha_config.get("base_url", "")
    user_id = str(interaction.user.id)

    # Use lock to prevent race conditions
    async with roll_lock:
        data = load_user_data()
        data = check_daily_reset(data)

        rolls_left = get_rolls_remaining(data, user_id)
        if rolls_left <= 0:
            await interaction.response.send_message("You've used all your rolls today! Win a battle for bonus rolls.")
            return

        available = get_available_characters(data, user_id)
        if not available:
            await interaction.response.send_message("No characters left in today's pool! Come back tomorrow.")
            return

        # Check for Quirk Chungus curse - forces QuirkMan roll
        user = get_user(data, user_id)
        quirk_curse = user.get("curses", {}).get("quirk_chungus", 0)
        curse_forced = False

        if quirk_curse > 0:
            quirk_char = get_char_by_id("QuirkMan")
            if quirk_char and data.get("daily_pool", {}).get("QuirkMan", 0) > 0:
                char = quirk_char
                curse_forced = True
                user["curses"]["quirk_chungus"] -= 1
                if user["curses"]["quirk_chungus"] <= 0:
                    del user["curses"]["quirk_chungus"]
            else:
                char = random.choice(available)
        else:
            char = random.choice(available)

        char_id = char.get("id", "unknown")
        char_name = char.get("name", "Unknown")
        rarity = char.get("rarity", "N")
        description = char.get("description", "A mysterious character.")

        use_roll(data, user_id)
        is_dupe = claim_character(data, user_id, char_id)
        save_user_data(data)

        user = get_user(data, user_id)
        collection = user.get("collection", {})
        copies = collection.get(char_id, 1)
        power_mult = get_power_multiplier(copies)
        unique_count = len(collection)
        total_count = len(gacha_config.get("characters", []))
        rolls_after = get_rolls_remaining(data, user_id)

    # Release lock before slow API call
    await interaction.response.defer()
    try:

        image_url = f"{base_url}/{char_id}-img.webp"

        prompt = f"""{interaction.user.display_name} just rolled a gacha and summoned this character:
Character: {char_name}
Rarity: {rarity}
Personality: {description}

Introduce this character to the user in 4-5 sentences. Be dramatic and playful. Address the user by name. Don't repeat the bio verbatim—tease the character's personality in your own words."""

        system_prompt = build_system_prompt("")
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        intro_text = response.choices[0].message.content
        intro_text = replace_placeholders(intro_text, interaction.user.display_name)

        emoji = RARITY_EMOJIS.get(rarity, "\u2753")
        color = RARITY_COLORS.get(rarity, 0x808080)

        embed = discord.Embed(
            title=f"{emoji} {rarity} — {char_name}",
            description=intro_text,
            color=color,
        )
        embed.set_image(url=image_url)

        # Footer shows power up info or NEW (and curse if forced)
        if curse_forced:
            footer_prefix = "🔮 CURSED ROLL!"
        elif is_dupe:
            footer_prefix = f"⬆️ POWER UP! x{copies} ({int(power_mult * 100)}% power)"
        else:
            footer_prefix = "✨ NEW!"
        embed.set_footer(text=f"{footer_prefix} • Collection: {unique_count}/{total_count} • Rolls left: {rolls_after}")

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"Something went wrong: {e}")


@bot.tree.command(name="collection", description="View your character collection")
async def slash_collection(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    data = load_user_data()
    data = check_daily_reset(data)
    user = get_user(data, user_id)

    collection = user.get("collection", {})
    # Handle old list format
    if isinstance(collection, list):
        collection = {cid: 1 for cid in collection}

    total_chars = len(gacha_config.get("characters", []))

    if not collection:
        await interaction.response.send_message("You haven't collected any characters yet! Use `/roll` to start.")
        return

    rarity_order = ["UR", "SSR", "SR", "R", "N"]
    grouped = {r: [] for r in rarity_order}

    for char_id, copies in collection.items():
        char = get_char_by_id(char_id)
        if char:
            rarity = char.get("rarity", "N")
            grouped[rarity].append((char, copies))

    unique_count = len(collection)
    total_copies = sum(collection.values())
    embed = discord.Embed(
        title=f"{interaction.user.display_name}'s Collection",
        description=f"**{unique_count}/{total_chars}** unique • **{total_copies}** total copies",
        color=0x9b59b6,
    )

    for rarity in rarity_order:
        chars = grouped[rarity]
        if chars:
            emoji = RARITY_EMOJIS.get(rarity, "")
            names = ", ".join(
                f"{c['name']} x{copies}" if copies > 1 else c["name"]
                for c, copies in chars
            )
            embed.add_field(name=f"{emoji} {rarity}", value=names, inline=False)

    rolls_left = get_rolls_remaining(data, user_id)
    embed.set_footer(text=f"Rolls remaining today: {rolls_left}")

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="pool", description="See today's available character pool")
async def slash_pool(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    data = load_user_data()
    data = check_daily_reset(data)

    pool = data.get("daily_pool", {})
    user = get_user(data, user_id)
    owned = set(user.get("collection", []))

    available_global = []
    available_for_user = []

    for char in gacha_config.get("characters", []):
        char_id = char["id"]
        if pool.get(char_id, 0) > 0:
            available_global.append(char)
            if char_id not in owned:
                available_for_user.append(char)

    embed = discord.Embed(
        title="Today's Character Pool",
        color=0x3498db,
    )

    if available_global:
        rarity_order = ["UR", "SSR", "SR", "R", "N"]
        grouped = {r: [] for r in rarity_order}
        for char in available_global:
            rarity = char.get("rarity", "N")
            name = char["name"]
            if char["id"] in owned:
                name = f"~~{name}~~"
            grouped[rarity].append(name)

        for rarity in rarity_order:
            names = grouped[rarity]
            if names:
                emoji = RARITY_EMOJIS.get(rarity, "")
                embed.add_field(name=f"{emoji} {rarity}", value=", ".join(names), inline=False)

        embed.description = f"**{len(available_global)}** characters in pool • **{len(available_for_user)}** available for you\n*(Strikethrough = already owned)*"
    else:
        embed.description = "The pool is empty! All characters have been claimed today."

    rolls_left = get_rolls_remaining(data, user_id)
    embed.set_footer(text=f"Your rolls remaining: {rolls_left}")

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="addchar", description="Add a character from Chub.ai to the gacha pool")
@app_commands.describe(
    url="Chub.ai character URL (e.g., https://chub.ai/characters/username/char-id)",
    rarity="Character rarity (default: SR)"
)
@app_commands.choices(rarity=[
    app_commands.Choice(name="N - Common", value="N"),
    app_commands.Choice(name="R - Rare", value="R"),
    app_commands.Choice(name="SR - Super Rare (default)", value="SR"),
    app_commands.Choice(name="SSR - Ultra Rare", value="SSR"),
    app_commands.Choice(name="UR - Legendary", value="UR"),
])
async def slash_addchar(
    interaction: discord.Interaction,
    url: str,
    rarity: str = "SR"
):
    await interaction.response.defer()

    # Parse URL
    parsed = parse_chub_url(url)
    if not parsed:
        await interaction.followup.send(
            "Invalid Chub.ai URL. Expected format: `https://chub.ai/characters/username/char-id`",
            ephemeral=True
        )
        return

    username, chub_char_id = parsed

    # Fetch and parse character
    try:
        card_data, image_bytes = await fetch_chub_character(username, chub_char_id)
    except Exception as e:
        await interaction.followup.send(f"Error fetching character: {e}", ephemeral=True)
        return

    if not card_data:
        await interaction.followup.send(
            "Could not fetch character from Chub.ai. Make sure the URL is correct and the character exists.",
            ephemeral=True
        )
        return

    # Convert to bot format with chosen rarity
    char = chub_to_bot_format(card_data, username, chub_char_id, rarity)

    # Check for duplicate
    existing = get_char_by_id(char["id"])
    if existing:
        await interaction.followup.send(
            f"A character with ID `{char['id']}` already exists! (Name: {existing['name']})",
            ephemeral=True
        )
        return

    # Upload image to R2
    try:
        upload_char_image_sync(char["id"], image_bytes)
    except Exception as e:
        await interaction.followup.send(f"Failed to upload image to storage: {e}", ephemeral=True)
        return

    # Add to gacha_config and save
    gacha_config["characters"].append(char)
    save_gacha_config()

    # Show confirmation
    emoji = RARITY_EMOJIS.get(rarity, "")
    color = RARITY_COLORS.get(rarity, 0x00ff00)

    embed = discord.Embed(
        title=f"{emoji} Added: {char['name']}",
        description=char.get("description", "")[:200] + "..." if len(char.get("description", "")) > 200 else char.get("description", ""),
        color=color
    )
    embed.set_thumbnail(url=f"{gacha_config['base_url']}/{char['id']}-img.webp")
    embed.add_field(name="Rarity", value=f"{emoji} {rarity}", inline=True)
    embed.add_field(name="Stats", value=f"HP: {char['stats']['hp']} | ATK: {char['stats']['atk']} | DEF: {char['stats']['def']}", inline=True)
    embed.add_field(name="Source", value=f"[Chub.ai]({char.get('chub_source', url)})", inline=False)
    embed.set_footer(text=f"Added by {interaction.user.display_name}")

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="battle", description="Battle another user's character")
@app_commands.describe(opponent="The user to battle")
async def slash_battle(interaction: discord.Interaction, opponent: discord.Member):
    user_id = str(interaction.user.id)
    opponent_id = str(opponent.id)

    if interaction.user.id == opponent.id:
        await interaction.response.send_message("You can't battle yourself!", ephemeral=True)
        return

    # Check if challenging the bot (Lilith)
    is_bot_battle = opponent.id == bot.user.id

    data = load_user_data()
    data = check_daily_reset(data)

    user1 = get_user(data, user_id)
    collection1 = user1.get("collection", {})

    # Handle old list format
    if isinstance(collection1, list):
        collection1 = {cid: 1 for cid in collection1}

    if not collection1:
        await interaction.response.send_message("You don't have any characters! Use `/roll` first.", ephemeral=True)
        return

    if not is_bot_battle:
        user2 = get_user(data, opponent_id)
        collection2 = user2.get("collection", {})
        if isinstance(collection2, list):
            collection2 = {cid: 1 for cid in collection2}
        if not collection2:
            await interaction.response.send_message(f"{opponent.display_name} doesn't have any characters yet!", ephemeral=True)
            return
    else:
        # Bot always uses Lilith at max power (x5)
        collection2 = {"Lilith": 5}

    await interaction.response.defer()
    try:
        char1_id = random.choice(list(collection1.keys()))
        char2_id = random.choice(list(collection2.keys()))
        char1 = get_char_by_id(char1_id)
        char2 = get_char_by_id(char2_id)

        if not char1 or not char2:
            await interaction.followup.send("Error loading character data.")
            return

        # Get base stats
        base_stats1 = char1.get("stats", {"hp": 50, "atk": 10, "def": 10})
        base_stats2 = char2.get("stats", {"hp": 50, "atk": 10, "def": 10})

        # Apply dupe power multiplier
        copies1 = collection1.get(char1_id, 1)
        copies2 = collection2.get(char2_id, 1)
        power1 = get_power_multiplier(copies1)
        power2 = get_power_multiplier(copies2)

        stats1 = {
            "hp": int(base_stats1["hp"] * power1),
            "atk": int(base_stats1["atk"] * power1),
            "def": int(base_stats1["def"] * power1)
        }
        stats2 = {
            "hp": int(base_stats2["hp"] * power2),
            "atk": int(base_stats2["atk"] * power2),
            "def": int(base_stats2["def"] * power2)
        }

        mod1 = round(random.uniform(0.8, 1.2), 2)
        mod2 = round(random.uniform(0.8, 1.2), 2)

        prompt = f"""You are a dramatic battle narrator. Two characters are fighting!

**{char1["name"]}** (fighting for {interaction.user.display_name})
- Stats: {stats1["hp"]} HP, {stats1["atk"]} ATK, {stats1["def"]} DEF
- Power Roll: {mod1}x multiplier
- Who they are: {char1.get("description", "Unknown")}

**{char2["name"]}** (fighting for {opponent.display_name})
- Stats: {stats2["hp"]} HP, {stats2["atk"]} ATK, {stats2["def"]} DEF
- Power Roll: {mod2}x multiplier
- Who they are: {char2.get("description", "Unknown")}

Write an exciting, dramatic battle scene (6-8 sentences). Describe the fight viscerally - their movements, attacks, reactions. Use their personalities to inform HOW they fight. Reference the power rolls affecting their performance. Build tension, then deliver a decisive conclusion.

The winner should be determined by: (ATK × Power Roll) vs opponent's DEF, with HP as tiebreaker. Higher effective damage wins.

End your narration with the victor standing triumphant. Then on the final line, write exactly:
WINNER: [owner's display name, either {interaction.user.display_name} or {opponent.display_name}]"""

        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        battle_text = response.choices[0].message.content

        winner_name = None
        winner_id = None
        lines = battle_text.strip().split("\n")
        for line in reversed(lines):
            if line.strip().upper().startswith("WINNER:"):
                winner_name = line.split(":", 1)[1].strip()
                break

        if winner_name:
            if winner_name.lower() == interaction.user.display_name.lower():
                winner_id = user_id
                winner_display = interaction.user.display_name
                winner_char = char1
            elif winner_name.lower() == opponent.display_name.lower():
                winner_id = opponent_id
                winner_display = opponent.display_name
                winner_char = char2
            else:
                score1 = (stats1["atk"] * mod1) - (stats2["def"] * 0.5) + (stats1["hp"] / 20)
                score2 = (stats2["atk"] * mod2) - (stats1["def"] * 0.5) + (stats2["hp"] / 20)
                if score1 >= score2:
                    winner_id = user_id
                    winner_display = interaction.user.display_name
                    winner_char = char1
                else:
                    winner_id = opponent_id
                    winner_display = opponent.display_name
                    winner_char = char2
        else:
            score1 = (stats1["atk"] * mod1) - (stats2["def"] * 0.5) + (stats1["hp"] / 20)
            score2 = (stats2["atk"] * mod2) - (stats1["def"] * 0.5) + (stats2["hp"] / 20)
            if score1 >= score2:
                winner_id = user_id
                winner_display = interaction.user.display_name
                winner_char = char1
            else:
                winner_id = opponent_id
                winner_display = opponent.display_name
                winner_char = char2

        # Grant bonus roll to winner (only if not the bot)
        if not is_bot_battle or winner_id == user_id:
            grant_bonus_roll(data, winner_id)
            save_user_data(data)
            footer_text = f"🏆 Winner: {winner_display} (+1 bonus roll)"
        else:
            # Bot won, no bonus roll
            footer_text = f"🏆 Winner: {winner_display} (Lilith claims victory!)"

        display_text = "\n".join(
            line for line in battle_text.split("\n")
            if not line.strip().upper().startswith("WINNER:")
        ).strip()

        embed = discord.Embed(
            title=f"⚔️ {char1['name']} vs {char2['name']}",
            description=display_text,
            color=RARITY_COLORS.get(winner_char.get("rarity", "N"), 0x808080),
        )
        # Show dupe level if > 1
        dupe_str1 = f" (x{copies1})" if copies1 > 1 else ""
        dupe_str2 = f" (x{copies2})" if copies2 > 1 else ""

        embed.add_field(
            name=f"{interaction.user.display_name}'s Fighter",
            value=f"**{char1['name']}**{dupe_str1}\nATK {stats1['atk']} • DEF {stats1['def']} • HP {stats1['hp']}\nRoll: {mod1}x",
            inline=True
        )
        embed.add_field(
            name=f"{opponent.display_name}'s Fighter",
            value=f"**{char2['name']}**{dupe_str2}\nATK {stats2['atk']} • DEF {stats2['def']} • HP {stats2['hp']}\nRoll: {mod2}x",
            inline=True
        )
        embed.set_footer(text=footer_text)

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"Battle error: {e}")


@bot.tree.command(name="harem", description="Watch your collected characters argue about who's your favorite")
async def slash_harem(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    data = load_user_data()
    data = check_daily_reset(data)
    user = get_user(data, user_id)

    collection = user.get("collection", {})
    # Handle old list format
    if isinstance(collection, list):
        collection = {cid: 1 for cid in collection}

    if len(collection) < 2:
        await interaction.response.send_message("You need at least 2 characters in your collection for a harem argument! Use `/roll` to collect more.")
        return

    await interaction.response.defer()
    try:
        char_list = []
        for char_id in collection.keys():
            char = get_char_by_id(char_id)
            if char:
                desc = get_char_description(char)
                char_list.append(f"**{char['name']}**: {desc}")

        chars_text = "\n\n".join(char_list)

        prompt = f"""You are writing a comedic scene. These characters are all in {interaction.user.display_name}'s collection and are arguing about who should be their favorite:

{chars_text}

Write a funny, chaotic argument scene (8-10 sentences) where they each make their case for why THEY should be the favorite and bicker with each other. Stay true to each character's personality and speech patterns. Make it dramatic, petty, and entertaining. They should interrupt each other, throw shade, and get increasingly unhinged."""

        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        scene_text = response.choices[0].message.content

        embed = discord.Embed(
            title=f"💕 {interaction.user.display_name}'s Harem Argument",
            description=scene_text,
            color=0xff69b4,
        )
        char_names = [get_char_by_id(cid)["name"] for cid in collection if get_char_by_id(cid)]
        embed.set_footer(text=f"Featuring: {', '.join(char_names)}")

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"Harem error: {e}")


@bot.tree.command(name="talk", description="Talk to a character in your collection")
@app_commands.describe(character="Character name to talk to", message="What you want to say")
async def slash_talk(interaction: discord.Interaction, character: str, message: str):
    user_id = str(interaction.user.id)
    data = load_user_data()
    data = check_daily_reset(data)
    user = get_user(data, user_id)
    collection = user.get("collection", {})

    # Handle old list format
    if isinstance(collection, list):
        collection = {cid: 1 for cid in collection}

    # Find the character
    char = find_char_by_name(character)

    if not char:
        available = [get_char_by_id(cid)["name"] for cid in collection if get_char_by_id(cid)]
        if available:
            await interaction.response.send_message(f"Character '{character}' not found. Your collection: {', '.join(available)}", ephemeral=True)
        else:
            await interaction.response.send_message(f"Character '{character}' not found and you have no characters! Use `/roll` to collect some.", ephemeral=True)
        return

    # Check if user owns this character (or allow Lilith as default)
    char_id = char["id"]
    if char_id not in collection and char_id != "Lilith":
        available = [get_char_by_id(cid)["name"] for cid in collection if get_char_by_id(cid)]
        if available:
            await interaction.response.send_message(f"You don't own **{char['name']}**! Your collection: {', '.join(available)}", ephemeral=True)
        else:
            await interaction.response.send_message(f"You don't own **{char['name']}**! Use `/roll` to collect characters.", ephemeral=True)
        return

    # Defer and get webhook
    await interaction.response.defer()
    base_url = gacha_config.get("base_url", "")
    avatar_url = f"{base_url}/{char_id}-img.webp" if base_url and char_id else None
    webhook = await webhook_cache.get_or_create(interaction.channel)

    try:
        # Fetch recent conversation context (include bot messages)
        context_messages = []
        async for msg in interaction.channel.history(limit=10):
            if msg.content and msg.id != interaction.id:
                author_name = msg.author.display_name
                context_messages.append(f"{author_name}: {msg.content}")
        context_messages.reverse()

        # Build the full prompt with context
        context_str = "\n".join(context_messages) if context_messages else ""
        if context_str:
            full_prompt = f"[Recent chat in this Discord channel:]\n{context_str}\n\n[{interaction.user.display_name} says to you:] {message}"
        else:
            full_prompt = f"[{interaction.user.display_name} says to you:] {message}"

        # Build character-specific system prompt
        char_prompt = build_gacha_char_prompt(char, interaction.user.display_name)
        system_prompt = f"{char_prompt}\n\n{MODE_INSTRUCTION_TALK}"

        response = await get_llm_response(system_prompt, full_prompt)
        response = response.replace("{{user}}", interaction.user.display_name).replace("{{USER}}", interaction.user.display_name)

        # Send via webhook (no header)
        if webhook:
            # Delete the deferred response
            await interaction.delete_original_response()
            if len(response) <= 2000:
                await webhook.send(content=response, username=char["name"], avatar_url=avatar_url)
            else:
                await webhook.send(content=response[:2000], username=char["name"], avatar_url=avatar_url)
                remaining = response[2000:]
                while remaining:
                    chunk = remaining[:2000]
                    remaining = remaining[2000:]
                    await webhook.send(content=chunk, username=char["name"], avatar_url=avatar_url)
        else:
            # Fallback if no webhook permission
            rarity = char.get("rarity", "N")
            emoji = RARITY_EMOJIS.get(rarity, "")
            header = f"**{emoji} {char['name']}:**\n"
            await interaction.followup.send(header + response)
    except Exception as e:
        await interaction.followup.send(f"Error talking to {char['name']}: {e}")


@bot.tree.command(name="curse", description="Sacrifice a character to curse another user")
@app_commands.describe(target="The user to curse")
async def slash_curse(interaction: discord.Interaction, target: discord.Member):
    user_id = str(interaction.user.id)
    target_id = str(target.id)

    if interaction.user.id == target.id:
        await interaction.response.send_message("You can't curse yourself! ...or can you? No, you can't.", ephemeral=True)
        return

    if target.bot and target.id != bot.user.id:
        await interaction.response.send_message("You can only curse humans... or Lilith.", ephemeral=True)
        return

    data = load_user_data()
    data = check_daily_reset(data)

    user = get_user(data, user_id)
    collection = user.get("collection", {})

    if not collection:
        await interaction.response.send_message("You need at least one character to sacrifice for a curse! Use `/roll` first.", ephemeral=True)
        return

    # Sacrifice a random character
    sacrificed_name, sacrifice_power, copies = sacrifice_character(data, user_id)

    # Pick random curse and apply with power scaling
    curse_type = random.choice(CURSE_TYPES)
    scale = apply_curse(data, target_id, curse_type, sacrifice_power)

    save_user_data(data)

    # Build response embed
    curse_name = get_curse_name(curse_type)
    curse_desc = get_curse_description(curse_type)
    dupe_str = f" x{copies}" if copies > 1 else ""

    embed = discord.Embed(
        title=f"🔮 {curse_name}",
        description=f"**{interaction.user.display_name}** sacrificed **{sacrificed_name}**{dupe_str} (power: {sacrifice_power}) to curse **{target.display_name}**!",
        color=0x800080,
    )
    embed.add_field(name="Effect", value=f"{curse_desc}\n*Scaled to {scale:.1f}x power!*", inline=False)

    # Show remaining curses on target
    target_user = get_user(data, target_id)
    active_curses = target_user.get("curses", {})
    if active_curses:
        curse_list = []
        for c_type, c_value in active_curses.items():
            curse_list.append(f"• {get_curse_name(c_type)}: {c_value} remaining")
        embed.add_field(name=f"{target.display_name}'s Active Curses", value="\n".join(curse_list), inline=False)

    await interaction.response.send_message(embed=embed)


@bot.event
async def on_message(message: discord.Message):
    # Don't let the bot respond to itself (non-webhook messages)
    if message.author.id == bot.user.id:
        return

    # All webhook messages (including our characters) can trigger other characters
    # The cooldown mechanism prevents infinite loops

    content = message.content.strip()

    # Check if sender is a character (to prevent self-triggering)
    sender_char = find_char_by_name(message.author.display_name)
    sender_id = sender_char["id"] if sender_char else None

    # Check if message contains character names (triggers character response)
    # Works for both users and bots - supports up to 3 characters per message
    matched_chars = []
    reply_triggered_chars = set()  # Track chars triggered by reply to prevent double-trigger

    # Bot-only chain safety cap: if 5+ consecutive bot messages, don't trigger more
    if message.webhook_id or message.author.bot:
        try:
            consecutive_bot_count = 1  # Current message is from a bot
            async for msg in message.channel.history(limit=5, before=message):
                if msg.webhook_id or msg.author.bot:
                    consecutive_bot_count += 1
                else:
                    break  # Human message found, stop counting
            if consecutive_bot_count >= 5:
                return  # Cap reached, don't trigger more bot responses
        except:
            pass

    # Check if this is a reply to a character's message
    if message.reference and message.reference.message_id:
        try:
            replied_msg = await message.channel.fetch_message(message.reference.message_id)
            replied_char = find_char_by_name(replied_msg.author.display_name)
            # Trigger the character being replied to (if not self)
            if replied_char and replied_char.get("id") != sender_id:
                matched_chars.append(replied_char)
                reply_triggered_chars.add(replied_char["id"])
        except:
            pass  # Couldn't fetch replied message

    # Strip punctuation for matching
    content_lower = content.lower()
    content_stripped = re.sub(r'[^\w\s]', '', content_lower)  # Remove punctuation

    # Special trigger words for specific characters
    special_triggers = {
        "locust": "Helena",
    }

    # Check special triggers first
    for trigger, trig_char_name in special_triggers.items():
        if trigger in content_stripped:
            char = find_char_by_name(trig_char_name)
            # Don't let character trigger themselves, and don't double-trigger from reply
            if char and char not in matched_chars and char.get("id") != sender_id and char.get("id") not in reply_triggered_chars:
                matched_chars.append(char)

    for c in gacha_config.get("characters", []):
        if c in matched_chars:
            continue  # Already added via special trigger or reply
        if c.get("id") == sender_id:
            continue  # Don't let character trigger themselves
        if c.get("id") in reply_triggered_chars:
            continue  # Already triggered by reply, prevent double-trigger
        name_lower = c["name"].lower()
        id_lower = c["id"].lower()
        name_stripped = re.sub(r'[^\w\s]', '', name_lower)
        id_stripped = re.sub(r'[^\w\s]', '', id_lower)

        # Check if name/id appears in the message (as whole word or substring)
        if name_stripped in content_stripped or id_stripped in content_stripped:
            matched_chars.append(c)
            if len(matched_chars) >= 3:
                break  # Max 3 characters

    if matched_chars:
        # Check cooldown - only block if 6+ triggers in last 10 seconds
        channel_id = message.channel.id
        now = time.time()

        # Get and clean up old timestamps
        if channel_id not in character_cooldowns:
            character_cooldowns[channel_id] = []
        character_cooldowns[channel_id] = [t for t in character_cooldowns[channel_id] if now - t < COOLDOWN_WINDOW_SECONDS]

        # Check if we're over the limit
        if len(character_cooldowns[channel_id]) >= COOLDOWN_MAX_TRIGGERS:
            return  # Too many triggers, skip

        # Record this trigger
        character_cooldowns[channel_id].append(now)

        # Queue up responses for each matched character
        async def queue_responses():
            for i, char in enumerate(matched_chars):
                if i > 0:
                    await asyncio.sleep(1)  # Small delay between responses
                await handle_name_trigger(message, char, content)

        asyncio.create_task(queue_responses())
        return

    # Bots can trigger slop detection but not commands
    if message.author.bot:
        # Check for slop keywords in bot messages
        if contains_slop_keyword(content):
            asyncio.create_task(handle_slop_trigger(message))
        return

    # Handle !ask command
    if content.startswith("!ask "):
        prompt = content[5:].strip()
        if prompt:
            asyncio.create_task(handle_ask(message, prompt))
        else:
            await message.reply("Please provide a prompt after `!ask`.")
        return

    # Handle !askcontext command - !askcontext [num] <prompt>
    if content.startswith("!askcontext "):
        args = content[12:].strip()
        if not args:
            await message.reply("Please provide a prompt after `!askcontext`.")
            return
        # Check if first word is a number
        parts = args.split(None, 1)
        if parts[0].isdigit():
            num_messages = min(int(parts[0]), 50)  # Cap at 50
            prompt = parts[1] if len(parts) > 1 else ""
        else:
            num_messages = 5
            prompt = args
        if prompt:
            asyncio.create_task(handle_askcontext(message, prompt, num_messages))
        else:
            await message.reply("Please provide a prompt after the number.")
        return

    # Handle !rewrite command
    if content.startswith("!rewrite "):
        instructions = content[9:].strip()
        if instructions:
            asyncio.create_task(handle_rewrite(message, instructions))
        else:
            await message.reply("Please provide rewrite instructions after `!rewrite`.")
        return

    # Handle !roll command
    if content == "!roll":
        asyncio.create_task(handle_roll(message))
        return

    # Handle !collection command
    if content == "!collection":
        asyncio.create_task(handle_collection(message))
        return

    # Handle !pool command
    if content == "!pool":
        asyncio.create_task(handle_pool(message))
        return

    # Handle !battle command
    if content.startswith("!battle "):
        # Extract mentioned user
        if message.mentions:
            opponent = message.mentions[0]
            asyncio.create_task(handle_battle(message, opponent))
        else:
            await message.reply("Please mention a user to battle: `!battle @user`")
        return

    # Handle !harem command
    if content == "!harem":
        asyncio.create_task(handle_harem(message))
        return

    # Handle !talk command - !talk <character> <message>
    if content.startswith("!talk "):
        args = content[6:].strip()
        if not args:
            await message.reply("Usage: `!talk <character> <message>`\nExample: `!talk Lilith hello there!`")
            return
        # Split into character name and message
        # First word (or quoted string) is character, rest is message
        if args.startswith('"'):
            # Quoted character name
            end_quote = args.find('"', 1)
            if end_quote != -1:
                char_name = args[1:end_quote]
                prompt = args[end_quote + 1:].strip()
            else:
                await message.reply("Missing closing quote for character name.")
                return
        else:
            # First word is character name
            parts = args.split(None, 1)
            char_name = parts[0]
            prompt = parts[1] if len(parts) > 1 else ""

        if not prompt:
            await message.reply("Please provide a message to send to the character.\nUsage: `!talk <character> <message>`")
            return

        asyncio.create_task(handle_talk(message, char_name, prompt))
        return

    # Handle !curse command
    if content.startswith("!curse "):
        if message.mentions:
            target = message.mentions[0]
            asyncio.create_task(handle_curse(message, target))
        else:
            await message.reply("Please mention a user to curse: `!curse @user`")
        return

    # Check for Lilith curse (20% chance to trigger slop rewrite)
    user_id = str(message.author.id)
    data = load_user_data()
    user = get_user(data, user_id)
    lilith_curse = user.get("curses", {}).get("lilith", 0)
    if lilith_curse > 0 and random.random() < 0.2:
        # Decrement curse counter
        user["curses"]["lilith"] -= 1
        if user["curses"]["lilith"] <= 0:
            del user["curses"]["lilith"]
        save_user_data(data)
        # Trigger slop rewrite
        asyncio.create_task(handle_slop_trigger(message, curse_triggered=True))
        return

    # Handle slop trigger (rewrite in character style)
    if contains_slop_keyword(content):
        asyncio.create_task(handle_slop_trigger(message))
        return

    # Handle keyword triggers (passive)
    if contains_trigger_keyword(content):
        asyncio.create_task(handle_keyword_trigger(message))
        return

    # Process other commands if any
    await bot.process_commands(message)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
