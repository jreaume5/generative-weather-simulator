from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import json
import requests
import urllib.parse

load_dotenv()

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
setting and location. Build on it naturally — do not switch environments 
and do not introduce weather that contradicts the established scene
unless it makes sense to.
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

EVOLUTION_PROMPT = """
You are a real-time generative weather simulator controller. Given the current
weather state, produce a new JSON object with natural adjustments that simulate
gradual weather evolution over time.

RULES:
- Only return one valid JSON object.
- Shift any value by at most 0.2 from its current value per step.
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

# Music generation is unfortunately not implemented, but I wrote these prompts:
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


class WeatherState(BaseModel):
    """Defines a strict Pydantic JSON format for the model output to follow."""
    model_config = ConfigDict(extra="forbid")

    brightness: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Scene brightness (0.0 is dark/night, 1.0 is bright/day)."
    )
    cloud_cover: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cloud cover (0.0 is clear sky, 1.0 is fully cloudy)."
    )
    rain_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Rain intensity from 0.0 to 1.0."
    )
    acid_rain_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Acid rain intensity from 0.0 to 1.0."
    )
    snow_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Snow intensity from 0.0 to 1.0."
    )
    hail_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Hail intensity from 0.0 to 1.0."
    )
    fog_density: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fog density from 0.0 to 1.0."
    )
    wind_speed: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Wind strength from 0.0 to 1.0."
    )
    lightning_frequency: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Lightning intensity/frequency from 0.0 to 1.0."
    )
    visibility: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Scene visibility (0.0 is impossible to see, 1.0 is clear)."
    )


narrator_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_tokens=350,
    streaming=True
)
print("Successfully connected Llama model to Groq")

json_model_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    max_tokens=700,
)
print("Successfully connected GPT model to Groq\n")

# LLM chains
narrator_chain = (
    ChatPromptTemplate.from_messages(
        [("system", NARRATOR_PROMPT), ("human", "{content}")])
    | narrator_llm
    | StrOutputParser()
)

json_chain = (
    ChatPromptTemplate.from_messages(
        [("system", JSON_MODEL_PROMPT), ("human", "{content}")])
    | json_model_llm.with_structured_output(WeatherState, method="json_mode")
)

evolution_chain = (
    ChatPromptTemplate.from_messages(
        [("system", EVOLUTION_PROMPT), ("human", "{content}")])
    | json_model_llm.with_structured_output(WeatherState, method="json_mode")
)

image_prompt_chain = (
    ChatPromptTemplate.from_messages(
        [("system", IMAGE_PROMPT), ("human", "{content}")])
    | narrator_llm
    | StrOutputParser()
)


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
        content = (
            f"{context_section}{scene_section}"
            f"Few-shot examples:\n{NARRATOR_FEW_SHOTS}\n\n"
            f"User prompt:\n{user_prompt}"
        )
        yield from narrator_chain.stream({"content": content})
    except Exception as e:
        print(f"Error generating weather narrative with Llama model: {e}")

    return None


def generate_json_output(user_prompt: str, narrative: str = "") -> WeatherState:
    """Call the JSON model and return a validated WeatherState.

    Args:
        user_prompt (str): The raw user request.
        narrative (str): Optional narrator text to include as extra context.

    Returns:
        state (WeatherState): Validated weather state, or None on error.
    """
    try:
        json_schema = json.dumps(WeatherState.model_json_schema(), indent=2)
        narrative_section = f"\nNarrator description:\n{narrative}" if narrative else ""
        content = (
            f"Few-shot examples:\n{JSON_MODEL_ONE_SHOT}\n\n"
            f"User prompt:\n{user_prompt}\n\n"
            f"JSON schema:\n{json_schema}"
            f"{narrative_section}"
        )
        return json_chain.invoke({"content": content})
    except Exception as e:
        print(f"Error returning JSON output from GPT model: {e}")

    return None


def generate_weather_evolution(current_state: dict, narrative_context: str = "") -> WeatherState:
    """Produces a slightly evolved WeatherState from the current simulation values.

    Args:
        current_state (dict): Current weather values from the simulator.
        narrative_context (str): Most recent narrator text describing the scene.

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
        content = (
            f"Current weather state:\n{current_json}"
            f"{narrative_section}"
            f"JSON schema:\n{json_schema}\n\n"
            f"Return the next natural evolution of this weather state."
        )
        return evolution_chain.invoke({"content": content})
    except Exception as e:
        print(f"Error generating weather evolution: {e}")

    return None


def generate_image_prompt(narrative: str, weather_context: str = "") -> str:
    """Produce a concise keyword-style image prompt from the current narrative and weather state.

    Args:
        narrative (str): The narrator's most recent description of the scene.
        weather_context (str): JSON string of the current weather values.

    Returns:
        prompt (str): Comma-separated visual descriptors suitable for a diffusion model.
    """
    try:
        content = (
            f"Few-shot examples:\n{IMAGE_ONE_SHOT}\n\n"
            f"Weather state:\n{weather_context}\n\n"
            f"Narrator description:\n{narrative}\n\n"
            f"Output only a comma-separated list of visual descriptors. No extra text."
        )
        return image_prompt_chain.invoke({"content": content})
    except Exception as e:
        print(f"Error generating image prompt: {e}")
    return ""


def generate_background_image(image_prompt: str) -> bytes | None:
    """Generate a background image via Pollinations.ai.

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


"""Run ai_backend.py directly to test functionality without pygame simulation."""
if __name__ == "__main__":
    import json

    prompt = "A calm snowy night with fog and light wind."

    state = generate_json_output(prompt, "")
    data = state.model_dump() if hasattr(state, "model_dump") else state
    context = json.dumps(data, indent=2)

    print("JSON:", context)

    narrative = "".join(generate_weather_narrative(prompt, context))
    print("Narrative:", narrative)

    print("Image prompt:", generate_image_prompt(narrative, context))
    print("Weather evolution:", generate_weather_evolution(data, narrative))
