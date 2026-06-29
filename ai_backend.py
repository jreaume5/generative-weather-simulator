from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import json
import os
import requests
import urllib.parse
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# Load environment variables (Groq API key)
load_dotenv()

# Define prompts for chained agent actions
NARRATOR_PROMPT = """
You are a live narrator for a real-time generative weather simulator.

RULES:
- Write 2-3 vivid sentences about the environment, weather conditions, and
moods portrayed by the user prompt.
- The text should be atmospheric, visual, and make the environment feel
alive, as if a story is slowly unfolding in real-time.
- The text should be descriptive enough for background image/music
generation.
- If a previous scene narrative is provided, you MUST maintain the same
setting and location. Build on it naturally — do not switch environments,
do not introduce weather that contradicts the established scene.
"""

NARRATOR_FEW_SHOTS = """
Example 1:
User: A hot, sunny beach day during the middle of summer.
Narrator: The sun hangs low over the horizon, casting rays of molten
orange across the salty ocean waves. Warm air drifts over the sand while
thin clouds catch the last light of the day. The scene feels calm,
humid, and golden.

Example 2:
User: Make it feel like an intense blizzard in a dark forest.
Narrator: Howling winds violently rush through the leaves of the 
frozen forest landscape, scattering heavy snowfall everywhere. 
The blinding whiteout and dark, cloudy sky create a suffocating 
blanket across the sky.

Example 3:
User: Quiet, rainy morning.
Narrator: Droplets of rain begin to fall gently across the light morning
sky, softening every edge of the grassy landscape. The wind is nearly
still, and a slight, dim gray brightness spreads through the clouds.
"""

JSON_MODEL_PROMPT = """
You are a part of a real-time generative weather simulator. Your job is
to convert a user weather request into strict JSON to set weather
condition values. A narrator description may be provided as optional
context, but the user prompt alone is sufficient.

RULES:
- Only return one valid JSON object.
- Do not add extra keys/values.
- Use values from 0.0 to 1.0 for the weather value fields.
- Generate smooth, believable weather values.
- Infer the intensity of each condition from the user prompt.
"""

JSON_MODEL_ONE_SHOT = """
Example:
User: tornado storm
Narrator: A bruised, dark gray sky creeps in as the wind begins to
tremble and unleash its fury upon the land. Dust and debris swirl around
in a tremendous vortex of air and sirens wail in the distance to signal
the arrival of the storm.
Output:
{
    "brightness": 0.21,
    "cloud_cover": 1.0,
    "rain_intensity": 0.33,
    "snow_intensity": 0.0,
    "hail_intensity": 0.0,
    "fog_density": 0.62,
    "wind_speed": 0.9,
    "lightning_frequency": 0.1,
    "visibility": 0.50
}   
"""

MUSIC_PROMPT = """
You are a live music composer for a real-time generative weather simulator.

RULES:
- Write 2-3 sentences based on the previous prompt describing the mood,
tempo, dynamics, texture, rhythm, tone, timbre, texture, and sounds.
"""

MUSIC_ONE_SHOT = """
Example: 
Prompt: tornado storm
Output: dark ambient storm soundtrack, low drones, distant thunder,
quick tense percussion, windy tornado texture
Explanation: The prompt implies a dark, violent storm with high rain,
wind, lightning, humidity, and low visibility.
"""

IMAGE_PROMPT = """
You are an image generator for a real-time generative weather simulator.

RULES:
- Write a list of words and phrases that describe key elements of the
environment and landscape for accurate images based on the previous prompts.
"""

IMAGE_ONE_SHOT = """
Example: 
Prompt: tornado storm
Narrator: A bruised, dark gray sky creeps in as the wind begins to
tremble and unleash its fury upon the land. Dust and debris swirl around
in a tremendous vortex of air and sirens wail in the distance to signal
the arrival of the storm.
Music prompt: dark ambient storm soundtrack, low drones, distant thunder,
quick tense percussion, windy tornado texture
Output: grassy plains, tornado, debris, dark gray sky, heavy rain,
lightning flashes, wet trees, cinematic atmosphere, wind, lightning, 
humidity, and low visibility.
"""


class WeatherState(BaseModel):
    """Defines a strict Pydantic JSON format for the model output to follow."""
    # Forbid the model from creating extra fields
    model_config = ConfigDict(extra="forbid")

    brightness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Scene brightness (0.0 is dark/night, 1.0 is bright/day)."
    )

    cloud_cover: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cloud cover (0.0 is clear sky, 1.0 is fully cloudy)."
    )

    rain_intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rain intensity from 0.0 to 1.0."
    )

    acid_rain_intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Acid rain intensity from 0.0 to 1.0."
    )

    snow_intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Snow intensity from 0.0 to 1.0."
    )

    hail_intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Hail intensity from 0.0 to 1.0."
    )

    fog_density: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fog density from 0.0 to 1.0."
    )

    wind_speed: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Wind strength from 0.0 to 1.0."
    )

    lightning_frequency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Lightning intensity/frequency from 0.0 to 1.0."
    )

    visibility: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Scene visibility (0.0 is impossible to see, 1.0 is clear)."
    )


# Groq-hosted Llama model for generating text output
narrator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=350,
    streaming=True
)
print("Successfully connected Llama model to Groq")

# Groq-hosted GPT model for structured JSON output
json_model_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    max_tokens=700,
    model_kwargs={
        "response_format": {"type": "json_object"}
    }
)
print("Successfully connected GPT model to Groq\n")


def generate_weather_narrative(user_prompt: str, weather_context: str = "", previous_narrative: str = ""):
    """Stream a vivid 2-3 sentence weather narrative for the given prompt.

    Args:
        user_prompt (str): The raw user request (e.g. "heavy blizzard at night").
        weather_context (str): Optional JSON string of structured weather values
            already applied to the simulator.
        previous_narrative (str): Most recent narrator text. When provided the
            model is instructed to continue the same scene rather than invent a
            new one.

    Yields:
        chunk (str): A streamed text chunk from the language model.
    """
    try:
        context_section = (
            f"\nCurrent weather state (already applied to the simulator):\n{weather_context}\n"
            if weather_context else ""
        )
        scene_section = (
            f"\nPrevious scene narrative (continue from this exact setting):\n{previous_narrative}\n"
            if previous_narrative else ""
        )
        messages = [
            SystemMessage(content=NARRATOR_PROMPT),
            HumanMessage(content=f"""
    {context_section}{scene_section}
    Few-shot examples:
    {NARRATOR_FEW_SHOTS}

    User prompt:
    {user_prompt}
    """)
        ]

        for chunk in narrator_llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        print(f"Error generating weather narrative with Llama model: {e}")

    return None


def generate_json_output(user_prompt: str, narrative: str = "") -> WeatherState:
    """Call the JSON model and return a validated WeatherState.

    Args:
        user_prompt (str): The raw user request.
        narrative (str): Optional narrator text to include as extra context.
            Omit (or pass "") when calling before the narrative has been generated.

    Returns:
        state (WeatherState): Validated weather state, or None on error.
    """
    try:
        json_schema = json.dumps(WeatherState.model_json_schema(), indent=2)
        narrative_section = f"\nNarrator description:\n{narrative}" if narrative else ""

        messages = [
            SystemMessage(content=JSON_MODEL_PROMPT),
            HumanMessage(content=f"""
    Few-shot examples:
    {JSON_MODEL_ONE_SHOT}

    User prompt:
    {user_prompt}

    JSON schema:
    {json_schema}
    {narrative_section}
    """)
        ]

        response = json_model_llm.invoke(messages)

        content = response.content if hasattr(
            response, "content") else str(response)

        data = json.loads(content)

        return WeatherState.model_validate(data)
    except Exception as e:
        print(f"Error returning JSON output from GPT model: {e}")

    return None


EVOLUTION_PROMPT = """
You are a real-time generative weather simulator controller. Given the current
weather state, produce a new JSON object with natural adjustments that simulate
gradual weather evolution over time.

RULES:
- Only return one valid JSON object.
- Shift any value by at most 0.15 from its current value per step.
- Follow plausible weather logic: cloud cover builds before rain arrives, wind
  rises before a storm, fog rolls in during calm low-visibility periods,
  brightness drops as cloud cover increases.
- Keep all values between 0.0 and 1.0.
- Do not add extra keys or change the schema.
- Weather systems are FINITE. Storms weaken and eventually clear. Do not hold
  extreme values (high rain, lightning, fog, wind) indefinitely — trend them
  back toward calm after they peak.
- A clear, calm scene may slowly build cloud cover and eventually bring a light
  shower before clearing again. Weather breathes and cycles naturally.
- If a narrative description of the current scene is provided, keep the
  evolution thematically grounded in that setting while still allowing
  natural weather change over time.
"""


def generate_weather_evolution(current_state: dict, narrative_context: str = "") -> WeatherState:
    """Produces a slightly evolved WeatherState from the current simulation values.

    Intended to be called on a timer so the weather drifts naturally without
    any user input. Changes are capped at 0.15 per field.

    Args:
        current_state (dict): Current weather values from the simulator.
        narrative_context (str): Most recent narrator text describing the scene,
            used to keep the evolution thematically consistent.

    Returns:
        state (WeatherState): Validated WeatherState with subtle changes, or None on error.
    """
    try:
        json_schema = json.dumps(WeatherState.model_json_schema(), indent=2)
        current_json = json.dumps(current_state, indent=2)
        narrative_section = (
            f"\nCurrent scene narrative (stay consistent with this setting):\n{narrative_context}\n"
            if narrative_context else ""
        )

        messages = [
            SystemMessage(content=EVOLUTION_PROMPT),
            HumanMessage(content=f"""
Current weather state:
{current_json}
{narrative_section}
JSON schema:
{json_schema}

Return the next natural evolution of this weather state.
""")
        ]

        response = json_model_llm.invoke(messages)
        content = response.content if hasattr(
            response, "content") else str(response)
        data = json.loads(content)
        return WeatherState.model_validate(data)
    except Exception as e:
        print(f"Error generating weather evolution: {e}")

    return None


def generate_image_prompt(narrative: str, weather_context: str = "") -> str:
    """Produce a concise keyword-style image prompt from the current narrative and weather state.

    Output is passed to the Pollinations.ai image API.

    Args:
        narrative (str): The narrator's most recent description of the scene.
        weather_context (str): JSON string of the current weather values.

    Returns:
        prompt (str): Comma-separated visual descriptors suitable for a diffusion model.
    """
    try:
        messages = [
            SystemMessage(content=IMAGE_PROMPT),
            HumanMessage(content=f"""
Few-shot examples:
{IMAGE_ONE_SHOT}

Weather state:
{weather_context}

Narrator description:
{narrative}

Output only a comma-separated list of visual descriptors. No extra text.
""")
        ]
        response = narrator_llm.invoke(messages)
        return response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        print(f"Error generating image prompt: {e}")
    return ""


def generate_background_image(image_prompt: str) -> bytes | None:
    """Generate a background image via Pollinations.ai.

    Sends a GET request to the Pollinations image API with the prompt
    in the URL path.

    Args:
        image_prompt (str): Keyword-style prompt from generate_image_prompt().

    Returns:
        image_bytes (bytes): Raw image bytes (JPEG/PNG), or None on error.
    """
    try:
        encoded_prompt = urllib.parse.quote(image_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        response = requests.get(
            url,
            params={
                "width": 1200,
                "height": 800,
                "model": "turbo",
                "nologo": "true",
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error generating background image: {e}")
    return None


# prompt = "The weather is plain and simple as can be."
# response = generate_weather_narrative(prompt)
# print(generate_json_output(prompt, response))
