# Lilith Discord Bot

A Discord bot with DeepSeek LLM integration featuring a gacha collection system, character battles, and curse mechanics.

## Features

### LLM Commands
- **!ask / /ask** - Ask the character a question
- **!askcontext [n] / /askcontext** - Ask with the last N messages as context
- **!rewrite / /rewrite** - Rewrite text in the character's style (reply to a message)
- **Right-click context menus** - "Ask about this" and "Rewrite this"

### Gacha System
- **!roll / /roll** - Roll for a random character (3 daily rolls)
- **!collection / /collection** - View your collected characters
- **!pool / /pool** - See today's available character pool

#### Rarity System
| Rarity | Daily Pool | Stats |
|--------|------------|-------|
| N | 5 copies | 50 HP, 10 ATK, 10 DEF |
| R | 3 copies | 70 HP, 15 ATK, 15 DEF |
| SR | 2 copies | 90 HP, 20 ATK, 20 DEF |
| SSR | 1 copy | 120 HP, 28 ATK, 25 DEF |
| UR | 1 copy | 150 HP, 35 ATK, 30 DEF |

#### Duplicate Power-Up System
Rolling a character you already own powers them up instead:
- Each duplicate adds +10% to all stats
- x2 copies = 110% power
- x3 copies = 120% power
- etc.

### Battle System
- **!battle @user / /battle** - Battle another user's character
- Winner receives a bonus roll
- Stats are modified by dupe power level and a random 0.8-1.2x multiplier
- You can challenge the bot itself (`@Lilith`) - she fights with x5 Lilith (140% power)

### Harem Mode
- **!harem / /harem** - Watch your collected characters argue about who's your favorite

### Curse System
- **!curse @user / /curse** - Sacrifice a character to curse another user
- Curse power scales with the sacrificed character's total stats (including dupe bonuses)

#### Curse Types
| Curse | Effect |
|-------|--------|
| Curse of Quirk Chungus | Adds QuirkMans to the pool + forces victim's next rolls to be QuirkMan |
| Curse of Lilith | 20% chance to trigger slop rewrite on victim's messages (max 25 triggers) |

### Passive Triggers
- **Slop Detection** - Messages containing LLM clichés ("delve", "testament to", "shivers down", etc.) get automatically rewritten
- **Help Triggers** - Messages containing phrases like "can someone explain" or "help me understand" trigger helpful interjections

## Setup

### Requirements
```
discord.py
openai
```

### Configuration
Edit the following in `bot.py`:

```python
DISCORD_TOKEN = "YOUR_DISCORD_TOKEN_HERE"
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
CHARACTER_CARD_PATH = "your_character_card.json"
```

### Character Card
The bot uses a v2 character card JSON file for the base character personality. This should include:
- `data.name` - Character name
- `data.description` - Character description
- `data.personality` - Personality traits
- `data.scenario` - Roleplay scenario (excluded from rewrite tasks)
- `data.creator_notes` - Additional guidance
- `data.extensions.depth_prompt.prompt` - Speech/behavior guidance
- `data.character_book.entries` - Lorebook entries

### Files
- `bot.py` - Main bot code
- `characters.json` - Gacha character definitions
- `user_data.json` - User collections and game state (auto-generated)
- `your_character_card.json` - Character personality card

### Running
```bash
python bot.py
```

## Character Data Format

Characters in `characters.json`:
```json
{
  "base_url": "https://your-cdn.com",
  "characters": [
    {
      "id": "CharacterName",
      "name": "Display Name",
      "rarity": "SR",
      "stats": {"hp": 90, "atk": 20, "def": 20},
      "description": "Character description for battle narration",
      "alt_description": "Optional extended description for harem mode"
    }
  ]
}
```

Character images should be hosted at `{base_url}/{id}-img.webp`.

## Daily Reset

- Pool refills at midnight (server time)
- Daily rolls reset to 3
- Bonus rolls from battles carry over
- Collections persist permanently

## Slop Keywords

The bot detects and rewrites messages containing common LLM clichés including:
- Purple prose ("searing kiss", "breath hitches", "ministrations")
- Overused phrases ("delve", "testament to", "vibrant tapestry")
- Dramatic clichés ("time stood still", "like a moth to a flame")
- Physical descriptions ("calloused hands", "wicked grin", "bit their lip")

See the full list in `SLOP_KEYWORDS` in `bot.py`.
