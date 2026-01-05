"""
Discord bot with DeepSeek LLM integration.
Supports direct commands, rewrite commands, and keyword triggers.
Uses v2 character card spec for base character/world context.
"""

import asyncio
import json
import random
import discord
from discord import app_commands
from discord.ext import commands
from openai import AsyncOpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

DISCORD_TOKEN = "DISCORD_TOKEN_HERE"
DEEPSEEK_API_KEY = "DEEPSEEK_KEY_HERE"
CHARACTER_CARD_PATH = "CHARACTER_JSON_FROM_CHUB_HERE"
GACHA_CONFIG_PATH = "characters.json"

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
Lilith rewrites the user's message in her unique style. Make it sloppy, and dripping with her personality. Output ONLY the rewritten text without any preamble or explanation. Do not respond to the message, just rewrite it. Keep it no more than 3 times the length of the original message."""

# =============================================================================
# CHARACTER CARD LOADING
# =============================================================================

def load_character_card(path: str) -> dict:
    """Load and parse a v2 character card JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_character_prompt(card: dict) -> str:
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

    if data.get("scenario"):
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

# =============================================================================
# BOT SETUP
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# DeepSeek client (OpenAI-compatible API)
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def build_system_prompt(mode_instruction: str) -> str:
    """Combine character base prompt with mode-specific instructions."""
    return f"{CHARACTER_BASE_PROMPT}\n\n{mode_instruction}"


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
    """Handle the !ask command."""
    async with message.channel.typing():
        try:
            system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK)
            response = await get_llm_response(system_prompt, prompt)
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

            system_prompt = build_system_prompt(MODE_INSTRUCTION_REWRITE)
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


async def handle_slop_trigger(message: discord.Message):
    """Handle slop trigger - rewrite user's message in character's sloppy style."""
    async with message.channel.typing():
        try:
            system_prompt = build_system_prompt(MODE_INSTRUCTION_SLOP)
            response = await get_llm_response(system_prompt, message.content)
            response = replace_placeholders(response, message.author.display_name)
            await send_long_message(message, response)
        except Exception as e:
            print(f"Slop trigger error: {e}")


async def handle_roll(message: discord.Message):
    """Handle the !roll gacha command."""
    characters = gacha_config.get("characters", [])
    base_url = gacha_config.get("base_url", "")

    if not characters:
        await message.reply("No characters configured for gacha.")
        return

    async with message.channel.typing():
        try:
            # Pick random character
            char = random.choice(characters)
            char_id = char.get("id", "unknown")
            char_name = char.get("name", "Unknown")
            rarity = char.get("rarity", "N")
            description = char.get("description", "A mysterious character.")

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
            embed.set_footer(text=f"Rolled by {message.author.display_name}")

            await message.reply(embed=embed)

        except Exception as e:
            await message.reply(f"Something went wrong: {e}")


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
        system_prompt = build_system_prompt(MODE_INSTRUCTION_ASK)
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


@bot.tree.command(name="rewrite", description="Rewrite text in character's style")
@app_commands.describe(text="The text to rewrite", instructions="How to rewrite it")
async def slash_rewrite(interaction: discord.Interaction, text: str, instructions: str):
    await interaction.response.defer()
    try:
        user_prompt = f"""Original text:
{text}

Rewrite instructions:
{instructions}"""
        system_prompt = build_system_prompt(MODE_INSTRUCTION_REWRITE)
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
    characters = gacha_config.get("characters", [])
    base_url = gacha_config.get("base_url", "")

    if not characters:
        await interaction.response.send_message("No characters configured for gacha.")
        return

    await interaction.response.defer()
    try:
        char = random.choice(characters)
        char_id = char.get("id", "unknown")
        char_name = char.get("name", "Unknown")
        rarity = char.get("rarity", "N")
        description = char.get("description", "A mysterious character.")

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
        embed.set_footer(text=f"Rolled by {interaction.user.display_name}")

        await interaction.followup.send(embed=embed)

    except Exception as e:
        await interaction.followup.send(f"Something went wrong: {e}")


@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from bots (including itself)
    if message.author.bot:
        return

    content = message.content.strip()

    # Handle !ask command
    if content.startswith("!ask "):
        prompt = content[5:].strip()
        if prompt:
            asyncio.create_task(handle_ask(message, prompt))
        else:
            await message.reply("Please provide a prompt after `!ask`.")
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
