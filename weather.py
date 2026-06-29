# --- CREDITS FOR BASELINE SIMULATION  ---
# Author: Dani Alc
# Mail: danialcdev@gmail.com
# GitHub Repository Link: https://github.com/DaniAlc0/weather
# This code can be used in any non-commerical project, and you may modify it as you want.

import pygame
import random
import math
import io
import os
import queue
import threading
import time
import json
from ai_backend import (generate_weather_narrative, generate_json_output,
                        generate_weather_evolution, generate_image_prompt,
                        generate_background_image)
pygame.mixer.init(frequency=44100, size=-16, buffer=2**12)
pygame.mixer.set_num_channels(24)


class Weather:

    def __init__(self, screen: pygame.Surface, weather_types: list[str] = None, wind_speed: int = 30, pixel: bool = False):
        """Initialize the Weather system.

        Args:
            screen (pygame.Surface): The Pygame screen where the weather effects will be drawn.
            weather_types (list[str]): List of weather effects to initialize. Supported types:
                'rain', 'acid rain', 'snow', 'hail', 'lightning', 'fog'.
            wind_speed (int): Maximum wind speed (ideally 0–100). Positive is right, negative is
                left. A value of exactly 0 disables wind and gusts entirely.
            pixel (bool): If True, snow and hail will be drawn as squares instead of circles.
        """
        self.screen = screen
        self.wind_speed = wind_speed
        self.wind = None
        self.pixel: bool = pixel
        # Dict that contains all current weather conditions. The key is a string and ther value the class itself
        self.effects: dict = {}

        # From 0.0 to 1.0: Volume modificator for all sounds
        self.general_vol: float = 1.0
        self.sounds: list = []  # Empty list to store all sounds before playing them
        # Empty dict to store all the channels playing sounds and their initial volume
        # Format: {channel: (effect_name, initial_volume)}
        self.channels: dict = {}
        # Track which mixer channels/sounds belong to each weather effect so
        # slider value 0 can stop that effect's audio immediately
        self.effect_channels: dict[str, list] = {}
        self.effect_sounds: dict[str, list[pygame.mixer.Sound]] = {}
        self.effect_intensities: dict[str, float] = {}
        self.rain_audio_sounds: list[pygame.mixer.Sound] = []
        self.rain_audio_channels: list[pygame.mixer.Channel] = []
        self.rain_audio_base_volume: float = 1.5 / 4
        self.wind_audio_sounds: list[pygame.mixer.Sound] = []
        self.wind_audio_channels: list[pygame.mixer.Channel] = []
        self.wind_audio_base_volume: float = 0.24
        # Used to store time for the delay playing different sounds
        self.timer: int = 0

        if weather_types is None:
            pass
        else:
            if wind_speed:
                self.wind = Wind(self, wind_speed, sound=True)

            if 'rain' in weather_types:
                self.effects['rain'] = Rain(self, screen, width=10, height=150, initial_speed=15, acc=50,
                                            color=(150, 200, 255, 155), flake=False, num_drops=30,
                                            sound_effect_name='rain')

            if 'acid rain' in weather_types:
                self.effects['acid rain'] = Rain(self, screen, width=10, height=150, initial_speed=15, acc=50,
                                                 color=(150, 255, 155, 155), flake=False, num_drops=30,
                                                 sound_effect_name='acid rain')

            if 'snow' in weather_types:
                self.effects['snow'] = Snow(self, screen, width=25, height=40, initial_speed=2, acc=1, color=(255, 255, 255, 255),
                                            flake=True, num_drops=40, pixel=self.pixel)

            if 'hail' in weather_types:
                self.effects['hail'] = Hail(self, screen, width=25, height=50, initial_speed=30, acc=20,
                                            color=(220, 220, 220, 220), flake=True, num_drops=30, pixel=self.pixel)

            if 'lightning' in weather_types:
                self.effects['lightning'] = Lightning(self, screen)

            if 'fog' in weather_types:
                self.effects['fog'] = Fog(screen, pixel=self.pixel)

        self.initial_sounds = self.sounds.copy()

    def update(self) -> None:
        """
        Update and render all active weather effects.
        """

        # Update wind and apply to other effects
        current_wind_speed = self.wind.update(
            self) if self.wind_speed and self.wind else 0

        # Update each other weather effect
        upd_rects = []

        for effect_name in ['lightning', 'fog', 'snow', 'rain', 'acid rain', 'hail']:
            if effect_name in self.effects:
                updated_rect = self.effects[effect_name].update(
                    current_wind_speed)
                if updated_rect is not None:
                    upd_rects.extend(updated_rect)

        # Play queued looping ambience sounds
        if self.sounds:
            new_timer = (pygame.time.get_ticks()//656 + 1) % 20
            if new_timer > self.timer:
                self.timer = new_timer

                queued_sound = None
                while self.sounds and queued_sound is None:
                    effect_name, sound = self._normalize_queued_sound(
                        self.sounds.pop(0))
                    if effect_name != 'unowned' and self.effect_intensities.get(effect_name, 1.0) <= 0.0:
                        continue
                    if effect_name != 'unowned' and effect_name not in self.effects:
                        continue
                    queued_sound = (effect_name, sound)

                if queued_sound is not None:
                    effect_name, sound = queued_sound
                    active_sound_count = max(
                        1, sum(len(sounds) for sounds in self.effect_sounds.values()))
                    volume = 1.5 / active_sound_count

                    channel = pygame.mixer.find_channel()  # Looks for an empty channel
                    if channel is not None:
                        # Plays sound on this channel in a loop
                        channel.play(sound, -1)

                        audio_scale = self._audio_scale(effect_name)
                        channel.set_volume(
                            volume * audio_scale * self.general_vol)
                        self.channels[channel] = (effect_name, volume)
                        self.effect_channels.setdefault(
                            effect_name, []).append(channel)
                # print(f'sec:{new_timer} queued weather sounds left: {len(self.sounds)}')

    def toggle_effect(self, effect_name: str) -> None:
        """Toggle the visibility of a specific weather effect."""
        if effect_name in self.effects:
            del self.effects[effect_name]
        else:
            self.__init__(
                self.screen, [effect_name] + list(self.effects.keys()))

    def set_wind_speed(self, base_max_speed, freq_base) -> None:
        """Manually set the wind speed."""
        self.wind.reset(self, base_max_speed, freq_base,
                        self.wind.max_gusts, self.wind.freq_gusts)

    def set_fog_density(self, density: float, color: list[int, int, int] | None = None) -> None:
        """Set the density (and color if needed) of the fog effect."""

        if 'fog' in self.effects:
            if color == None:
                color = self.effects['fog'].color
            self.effects['fog'].load_images(density, color, self.pixel)

    def set_lightning_frequency(self, frequency: int) -> None:
        """Set the frequency of lightning strikes.

        Args:
            frequency (int): Milliseconds between strikes (+/- 20%).
        """

        if 'lightning' in self.effects:
            self.effects['lightning'].frequency = frequency

    def change_volume(self, new_vol):
        """Set the master volume for all active sounds.

        Args:
            new_vol (float): Volume level from 0.0 (silent) to 1.0 (full initial volume).

        Raises:
            ValueError: If new_vol is outside the range [0.0, 1.0].
        """

        if 0.0 <= new_vol <= 1.0:
            self.general_vol = new_vol
            for channel, channel_info in list(self.channels.items()):
                if isinstance(channel_info, tuple):
                    effect_name, ini_vol = channel_info
                else:
                    effect_name, ini_vol = 'unowned', channel_info
                audio_scale = self._audio_scale(effect_name)
                channel.set_volume(ini_vol * audio_scale * new_vol)

            self._set_rain_audio_volume()
            self._set_wind_audio_volume()

            if 'lightning' in self.effects.keys():
                self.effects['lightning'].general_vol = new_vol

        else:
            raise ValueError

    def _combined_rain_audio_intensity(self) -> float:
        """
        Rain and acid rain share the same looping rain sound files.
        Combine their slider values into one rain-audio bus so one slider
        cannot stop the other slider's ambience.
        """
        rain = self.effect_intensities.get('rain', 0.0)
        acid = self.effect_intensities.get('acid rain', 0.0)
        return max(0.0, min(1.0, rain + acid))

    def _rain_audio_scale(self) -> float:
        intensity = self._combined_rain_audio_intensity()
        if intensity <= 0.0:
            return 0.0
        return math.sqrt(intensity)

    def _ensure_rain_audio_started(self) -> None:
        """Start the shared rain/acid-rain ambience bus if it is needed."""
        if self._combined_rain_audio_intensity() <= 0.0:
            self._stop_rain_audio()
            return

        active_channels = [
            ch for ch in self.rain_audio_channels if ch.get_busy()]
        if len(active_channels) == len(self.rain_audio_sounds) and active_channels:
            self.rain_audio_channels = active_channels
            return

        # Rebuild bus if it was stopped or partially lost
        self._stop_rain_audio()
        self.rain_audio_sounds = [
            pygame.mixer.Sound(f'assets/weather/rain/{i+1}.mp3')
            for i in range(4)
        ]
        self.rain_audio_channels = []

        for sound in self.rain_audio_sounds:
            channel = pygame.mixer.find_channel()
            if channel is None:
                break
            channel.play(sound, -1)
            self.rain_audio_channels.append(channel)

        if self.rain_audio_channels:
            self._set_rain_audio_volume()

    def _set_rain_audio_volume(self) -> None:
        """Apply combined rain/acid-rain slider volume to the shared rain bus."""
        scale = self._rain_audio_scale()
        if scale <= 0.0:
            self._stop_rain_audio()
            return

        if not self.rain_audio_channels:
            if not self.rain_audio_sounds:
                self._ensure_rain_audio_started()
            return

        volume = self.rain_audio_base_volume * scale * self.general_vol
        for channel in list(self.rain_audio_channels):
            if channel.get_busy():
                channel.set_volume(volume)
            else:
                self.rain_audio_channels.remove(channel)

        if not self.rain_audio_channels and not self.rain_audio_sounds:
            self._ensure_rain_audio_started()

    def _stop_rain_audio(self) -> None:
        """Stop the shared rain/acid-rain ambience bus only when both are 0%."""
        for channel in list(self.rain_audio_channels):
            channel.stop()
        self.rain_audio_channels.clear()

        for sound in list(self.rain_audio_sounds):
            sound.stop()
        self.rain_audio_sounds.clear()

    def _wind_audio_scale(self) -> float:
        intensity = max(
            0.0, min(1.0, self.effect_intensities.get('wind', 0.0)))
        if intensity <= 0.0:
            return 0.0

        return 0.20 + 0.80 * math.sqrt(intensity)

    def _ensure_wind_audio_started(self) -> None:
        """Start/rebuild the wind ambience loop when the wind slider is above 0%."""
        if self.effect_intensities.get('wind', 0.0) <= 0.0:
            self._stop_wind_audio()
            return

        active_channels = [
            ch for ch in self.wind_audio_channels if ch.get_busy()]
        if len(active_channels) == len(self.wind_audio_sounds) and active_channels:
            self.wind_audio_channels = active_channels
            return

        self._stop_wind_audio()
        self.wind_audio_sounds = [
            pygame.mixer.Sound(f'assets/weather/wind/{i+1}.mp3')
            for i in range(3)
        ]
        self.wind_audio_channels = []

        for sound in self.wind_audio_sounds:
            channel = pygame.mixer.find_channel()
            if channel is None:
                break
            channel.play(sound, -1)
            self.wind_audio_channels.append(channel)

        self._set_wind_audio_volume()

    def _set_wind_audio_volume(self) -> None:
        """Apply wind slider + master volume to the stable wind audio bus."""
        scale = self._wind_audio_scale()
        if scale <= 0.0:
            self._stop_wind_audio()
            return

        if not self.wind_audio_channels:
            self._ensure_wind_audio_started()
            return

        volume = self.wind_audio_base_volume * scale * self.general_vol
        for channel in list(self.wind_audio_channels):
            if channel.get_busy():
                channel.set_volume(volume)
            else:
                self.wind_audio_channels.remove(channel)

        # If every wind channel died unexpectedly, rebuild on the next slider/update call.
        if not self.wind_audio_channels and self.effect_intensities.get('wind', 0.0) > 0.0:
            self._ensure_wind_audio_started()

    def _stop_wind_audio(self) -> None:
        """Stop only wind-owned audio without touching rain/acid-rain/hail sounds."""
        for channel in list(self.wind_audio_channels):
            channel.stop()
        self.wind_audio_channels.clear()

        for sound in list(self.wind_audio_sounds):
            sound.stop()
        self.wind_audio_sounds.clear()

    def queue_effect_sound(self, effect_name: str, sound: pygame.mixer.Sound) -> None:
        """Queue a looping ambience sound and remember which effect owns it."""
        self.effect_sounds.setdefault(effect_name, []).append(sound)
        self.sounds.append((effect_name, sound))

    def register_effect_sound(self, effect_name: str, sound: pygame.mixer.Sound) -> None:
        """Remember a one-shot sound so it can be stopped if the effect is disabled."""
        self.effect_sounds.setdefault(effect_name, []).append(sound)

    def _normalize_queued_sound(self, queued_item):
        """Support both new (effect_name, sound) tuples and old bare Sound objects."""
        if isinstance(queued_item, tuple) and len(queued_item) == 2:
            return queued_item
        return 'unowned', queued_item

    def _audio_scale(self, effect_name: str) -> float:
        """
        Convert a visual slider value to an audio multiplier.

        Visual intensity remains linear so particle counts still feel responsive,
        but audio uses a square-root curve so it does not become too quiet too
        quickly. Examples: 25% slider -> 50% audio, 50% slider -> 71% audio.
        A true 0 still disables and stops the effect.
        """
        intensity = max(
            0.0, min(1.0, self.effect_intensities.get(effect_name, 1.0)))
        if intensity <= 0.0:
            return 0.0
        return math.sqrt(intensity)

    def _set_effect_audio_volume(self, effect_name: str) -> None:
        """Apply the current slider intensity and master volume to active channels."""
        audio_scale = self._audio_scale(effect_name)
        for channel in list(self.effect_channels.get(effect_name, [])):
            if channel in self.channels:
                _, ini_vol = self.channels[channel]
                channel.set_volume(ini_vol * audio_scale * self.general_vol)

    def _stop_effect_audio(self, effect_name: str) -> None:
        """Stop queued and currently playing sounds for one weather effect."""
        kept_sounds = []
        for queued_item in self.sounds:
            queued_effect_name, _ = self._normalize_queued_sound(queued_item)
            if queued_effect_name != effect_name:
                kept_sounds.append(queued_item)
        self.sounds = kept_sounds

        for channel in list(self.effect_channels.get(effect_name, [])):
            channel.stop()
            self.channels.pop(channel, None)
        self.effect_channels.pop(effect_name, None)

        for sound in self.effect_sounds.get(effect_name, []):
            sound.stop()
        self.effect_sounds.pop(effect_name, None)

    def add_effect(self, effect_name: str) -> None:
        """Create a weather effect without reinitializing the whole Weather object."""
        if effect_name in self.effects:
            return

        old_sound_count = len(self.sounds)

        if effect_name == 'rain':
            self.effects['rain'] = Rain(self, self.screen, width=10, height=150, initial_speed=15, acc=50,
                                        color=(150, 200, 255, 155), flake=False, num_drops=30,
                                        sound_effect_name='rain')
        elif effect_name == 'acid rain':
            self.effects['acid rain'] = Rain(self, self.screen, width=10, height=150, initial_speed=15, acc=50,
                                             color=(150, 255, 155, 155), flake=False, num_drops=30,
                                             sound_effect_name='acid rain')
        elif effect_name == 'snow':
            self.effects['snow'] = Snow(self, self.screen, width=25, height=40, initial_speed=2, acc=1,
                                        color=(255, 255, 255, 255), flake=True, num_drops=40, pixel=self.pixel)
        elif effect_name == 'hail':
            self.effects['hail'] = Hail(self, self.screen, width=25, height=50, initial_speed=30, acc=20,
                                        color=(220, 220, 220, 220), flake=True, num_drops=30, pixel=self.pixel)
        elif effect_name == 'lightning':
            self.effects['lightning'] = Lightning(self, self.screen)
        elif effect_name == 'fog':
            self.effects['fog'] = Fog(self.screen, pixel=self.pixel)
        else:
            raise ValueError(f"Unsupported weather effect: {effect_name}")

        if len(self.sounds) > old_sound_count:
            self.initial_sounds.extend(self.sounds[old_sound_count:])

    def remove_effect(self, effect_name: str) -> None:
        """Remove a weather effect from rendering and stop its audio."""
        if effect_name in self.effects:
            del self.effects[effect_name]
        self.effect_intensities[effect_name] = 0.0
        self._stop_effect_audio(effect_name)
        if effect_name in ('rain', 'acid rain'):
            self._set_rain_audio_volume()

    def set_wind_intensity(self, intensity: float) -> None:
        """
        Slider-friendly wind control.
        intensity 0.0 disables wind; 1.0 creates strong gusting wind.
        """
        intensity = max(0.0, min(1.0, float(intensity)))
        self.effect_intensities['wind'] = intensity

        if intensity <= 0.0:
            self.wind_speed = 0
            self._stop_wind_audio()
            self.wind = None
            return

        self.wind_speed = int(5 + 95 * intensity)

        if self.wind is None:
            self.wind = Wind(self, self.wind_speed, amplitude=55,
                             freq_base=50, max_gusts=3, freq_gusts=50, sound=False)
        else:
            self.wind.base_max_speed = self.wind_speed
            self.wind.amplitude = int(20 + 70 * intensity)
            self.wind.freq_base = int(15 + 85 * intensity)
            self.wind.max_gusts = int(1 + 9 * intensity)
            self.wind.freq_gusts = int(15 + 85 * intensity)

        self._ensure_wind_audio_started()
        self._set_wind_audio_volume()

    def set_effect_intensity(self, effect_name: str, intensity: float) -> None:
        """
        Set a weather condition using a normalized slider value from 0.0 to 1.0.
        A value of 0 disables the effect. Higher values increase visual intensity.
        """
        intensity = max(0.0, min(1.0, float(intensity)))

        if effect_name == 'wind':
            self.set_wind_intensity(intensity)
            return

        if effect_name == 'volume':
            self.change_volume(intensity)
            return

        self.effect_intensities[effect_name] = intensity

        if intensity <= 0.0:
            self.remove_effect(effect_name)
            return

        if effect_name not in self.effects:
            self.add_effect(effect_name)

        if effect_name in ('rain', 'acid rain'):
            self._ensure_rain_audio_started()
            self._set_rain_audio_volume()
        else:
            self._set_effect_audio_volume(effect_name)

        effect = self.effects[effect_name]

        if effect_name == 'rain':
            effect.num_drops = int(5 + 115 * intensity)
            effect.initial_speed = 8 + 22 * intensity
            effect.acc = 10 + 70 * intensity
            effect.color = (150, 200, 255, int(80 + 175 * intensity))

        elif effect_name == 'acid rain':
            effect.num_drops = int(5 + 115 * intensity)
            effect.initial_speed = 8 + 22 * intensity
            effect.acc = 10 + 70 * intensity
            effect.color = (150, 255, 155, int(80 + 175 * intensity))

        elif effect_name == 'snow':
            effect.num_drops = int(5 + 135 * intensity)
            effect.initial_speed = 0.5 + 5.5 * intensity
            effect.acc = 0.1 + 3.0 * intensity
            effect.color = (255, 255, 255, int(120 + 135 * intensity))

        elif effect_name == 'hail':
            effect.num_drops = int(2 + 78 * intensity)
            effect.initial_speed = 10 + 40 * intensity
            effect.acc = 5 + 40 * intensity
            effect.color = (220, 220, 220, int(120 + 135 * intensity))

        elif effect_name == 'lightning':
            # Lower milliseconds means more frequent lightning.
            effect.frequency = int(12000 - 11500 * intensity)
            effect.frequency = max(500, effect.frequency)
            effect.effect_vol = self._audio_scale('lightning')

        elif effect_name == 'fog':
            self.set_fog_density(intensity)

        if effect_name in ('rain', 'acid rain'):
            self._set_rain_audio_volume()
        else:
            self._set_effect_audio_volume(effect_name)

    def get_slider_value(self, sliders, effect_name, default=0.0):
        """Find a slider by effect_name and return its current value."""
        for slider in sliders:
            if slider.effect_name == effect_name:
                return slider.value

        return default

    def get_weather_effect_values(self, sliders, app_state):
        """Returns a json structure containing the slider values for
        each weather state."""
        weather_states = {
            "brightness": app_state["brightness"],
            "cloud_cover": app_state["cloud_cover"],
            "rain_intensity": self.get_slider_value(sliders, "rain"),
            "acid_rain_intensity": self.get_slider_value(sliders, "acid rain"),
            "snow_intensity": self.get_slider_value(sliders, "snow"),
            "hail_intensity": self.get_slider_value(sliders, "hail"),
            "fog_density": self.get_slider_value(sliders, "fog"),
            "wind_speed": self.get_slider_value(sliders, "wind"),
            "lightning_frequency": self.get_slider_value(sliders, "lightning"),
            "visibility": app_state["visibility"]
        }
        return weather_states


class Wind:
    """Represents wind, which affects the movement of precipitation.

    Args:
        weather (Weather): Instance of the Weather class.
        base_max_speed (int): Maximum base wind speed (ideally -100 to 100); higher absolute
            values produce stronger wind.
        amplitude (int): Oscillation range from 0 to 100. 0 keeps speed constant; 100 lets it
            swing between -base_max_speed and +base_max_speed.
        freq_base (int): Frequency of base wind variation (ideally 0–100; above 100 is erratic).
        max_gusts (int): Maximum additional speed during gusts.
        freq_gusts (int): Frequency of gusts (ideally 0–100).
        sound (bool): If True, wind sounds will be generated.
    """

    def __init__(self, weather,  base_max_speed: int, amplitude: int = 55, freq_base: int = 50, max_gusts: int = 3, freq_gusts: int = 50, sound: bool = True):
        self.reset(weather, base_max_speed, amplitude,
                   freq_base, max_gusts, freq_gusts, sound)

    def reset(self, weather, base_max_speed, amplitude, freq_base, max_gusts, freq_gusts, sound):
        self.base_max_speed: int = base_max_speed
        self.freq_base: int = freq_base
        self.amplitude: int = amplitude
        self.max_gusts: int = max_gusts
        self.gusts = random.uniform(self.max_gusts / 2, self.max_gusts)
        self.freq_gusts: int = freq_gusts
        self.sound: bool = sound

        self.wind_sounds = []
        self.w_timer = 0  # Used to store time for the delay playing different wind sounds

        self.start_time = pygame.time.get_ticks()

        if sound:
            for i in range(3):
                sound = pygame.mixer.Sound(f'assets/weather/wind/{i+1}.mp3')
                self.wind_sounds.append(sound)
                channel = pygame.mixer.find_channel()
                if channel is not None:
                    channel.play(self.wind_sounds[i], -1)

        # Update speed at init
        self.update(weather)

    def update(self, weather) -> float:
        """Update wind speed based on sinusoidal base variation and random gusts.

        Args:
            weather (Weather): The parent Weather instance (used for audio scaling).

        Returns:
            speed (float): Current instantaneous wind speed.
        """
        t = (pygame.time.get_ticks() - self.start_time) / \
            1000  # Time in seconds

        # Generate a sinusoidal value for regular wind
        max_speed = self.base_max_speed
        dif_speed = (self.base_max_speed * self.amplitude) // 100
        mean_speed = max_speed - dif_speed

        base_var = dif_speed*math.sin(2 * math.pi * self.freq_base/1000 * t)

        base_wind = mean_speed+base_var

        # Generate a series of sinusoidal gusts
        gusts_sum = 0
        for i in range(10):
            gusts_var = math.sin(2 * math.pi * (i+1) * self.freq_gusts/500 * t)
            gusts_sum += self.gusts * gusts_var

        # When gusts value is close to 0, a new base speed is established the gusts
        if -0.01 < gusts_var < 0.01:
            self.gusts = random.uniform(self.max_gusts // 2, self.max_gusts)

        # Add the base value and the gust to get the total speed
        self.speed = base_wind + gusts_sum

        if self.sound:
            for i in range(len(self.wind_sounds)):
                wind_audio_scale = weather._wind_audio_scale()
                self.wind_sounds[i].set_volume(
                    max(0.0, 0.18 * wind_audio_scale * weather.general_vol))

        return self.speed


class Precipitation:
    """Base class for precipitation effects such as rain, snow, or hail.

    Drop size, height, and initial speed follow a linear scale ranging from
    25% of the max value for the smallest drop to 100% for the largest.

    Args:
        weather (Weather): Instance of the Weather class.
        screen (pygame.Surface): The Pygame screen where the effects will be drawn.
        width (int): Max width of each drop.
        height (int): Max height of each drop.
        initial_speed (int): Max initial fall speed for each drop.
        acc (int): Max acceleration for each drop.
        color (tuple[int, int, int, int]): RGBA color of the drops.
        num_drops (int): Number of drops maintained on screen at all times.
        flake (bool): If True, a circle (or square when pixel=True) is drawn at the
            bottom of each sprite.
        pixel (bool): If True, the flake is drawn as a square instead of a circle.
    """

    def __init__(self, weather, screen: pygame.Surface, width: int, height: int, initial_speed: int, acc: int,
                 color: tuple[int, int, int, int], num_drops: int, flake: bool = False, pixel: bool = False, is_hail=False):
        self.screen = screen
        self.width = width
        self.height = height
        self.initial_speed = initial_speed
        self.acc = acc
        self.flake = flake
        self.is_hail = is_hail
        self.color = color
        self.pixel = pixel

        self.num_drops = num_drops
        self.drops = []  # List to store all the drops from the current class

    def create_drop(self, screen, initial_speed, acc, flake, is_hail):

        scale = 0.35 + 0.65 * random.random()
        weight = scale*0.9
        w, h = int(scale * self.width), int(scale * self.height)
        speed = scale * initial_speed
        acceleration = scale * acc / 100  # The bigger the more it accelerates

        pic = pygame.Surface((w, h), pygame.SRCALPHA, 32).convert_alpha()
        r, g, b, a = self.color

        transparency_tail_x = [
            1 - abs((i - w // 2) / (w // 2)) for i in range(w)]
        transparency_tail_y = (a * scale) / h

        for j in range(h - w):
            for i in range(w):
                alphax = transparency_tail_x[i] ** 2
                alphay = transparency_tail_y * j
                rect = (i, j, 1, 1)  # Pixel in position i,j
                pic.fill((r, g, b, int(alphay * alphax)), rect)

        if flake:
            weight /= 2
            if self.pixel:
                pygame.draw.rect(pic, (r, g, b, 255),
                                 (w // 4, h - w, w // 2, w // 2))
            else:
                pygame.draw.circle(pic, (r, g, b, 255),
                                   (w // 2, h - w), w // 4)

        if is_hail:
            new_drop = Hail.Drop(speed, acceleration, weight, pic, screen)
        else:
            new_drop = Precipitation.Drop(
                speed, acceleration, weight, pic, screen)

        self.drops.append(new_drop)

    def update(self, wind_speed: float = 0) -> list[pygame.Rect]:
        """Update and render all precipitation drops.

        Args:
            wind_speed (float): Current wind speed affecting the precipitation.

        Returns:
            update_rects (list[pygame.Rect]): Dirty rectangles that need to be redrawn.
        """

        # Add or delete drops until there are the same as self.num_drops
        if len(self.drops) < self.num_drops:
            self.create_drop(self.screen, self.initial_speed,
                             self.acc, self.flake, self.is_hail)
        elif len(self.drops) > self.num_drops:
            del self.drops[0]

        # Update the rectangles where the drops are now and where they were in the last loop
        update_rects = []
        for drop in self.drops:
            r = drop.render(self.screen, wind_speed)
            if r:
                i = r.collidelist(update_rects)
                if i > -1:
                    update_rects[i].union_ip(r)
                else:
                    update_rects.append(r)
        return update_rects

    class Drop:
        """A single drop used by the precipitation generator."""

        def __init__(self, speed: float, acc: float, weight: float, pic: pygame.Surface, screen: pygame.Surface):
            """Initialize a precipitation drop.

            Args:
                speed (float): Initial fall speed of the drop.
                acc (float): Downward acceleration per frame.
                weight (float): Mass factor from 0 to 1; higher values resist wind deflection.
                pic (pygame.Surface): Pre-rendered surface representing the drop.
                screen (pygame.Surface): The Pygame screen where the drop will be drawn.
            """
            self.pic = pic
            self.size = pic.get_size()

            self.ini_speed = speed
            self.acceleration = acc
            self.weight = weight

            self.screen_w = screen.get_width()
            self.screen_h = screen.get_height()

            self.pos = [random.random() * self.screen_w, -
                        random.randint(-self.screen_h, self.screen_h)]
            self.current_speed_x = 0
            self.current_speed_y = self.ini_speed * random.uniform(1, 1.5)

        def _reset_on_top(self, wind_speed) -> None:
            """Restart the drop at the top of the screen."""
            self.current_speed_y = self.ini_speed * random.uniform(1, 1.5)
            self.current_speed_x = self.current_speed_x//2 + wind_speed//4
            self.pos = [random.random() * self.screen_w, - self.size[1]]

        def _reset_on_sides(self, left: bool) -> None:
            """Restart the drop on one side of the screen."""
            self.current_speed_y = self.ini_speed * random.uniform(1, 1.5)
            if left:
                self.pos = [-self.size[0], random.random() * self.screen_h]
            else:
                self.pos = [self.screen_w, random.random() * self.screen_h]

        def render(self, screen: pygame.Surface, wind_speed: float) -> pygame.Rect | None:
            """Update the drop's position and speed, then draw it on the screen.

            Args:
                screen (pygame.Surface): The Pygame screen where the drop will be drawn.
                wind_speed (float): Current wind speed affecting horizontal drift.

            Returns:
                rect (pygame.Rect): The bounding rectangle covering old and new positions,
                    or None if the drop was not drawn.
            """

            oldrect = self.pic.get_rect()

            rotated_pic = self.pic  # Initialize to the default picture

            if wind_speed:
                dx = (wind_speed - self.current_speed_x)*(1 - self.weight)**2

                self.current_speed_x += dx/1000
                if self.current_speed_x > 1:
                    self.current_speed_x -= (self.current_speed_x**0.5) / 100
                elif self.current_speed_x < -1:
                    self.current_speed_x += ((abs(self.current_speed_x))
                                             ** 0.5) / 100
                self.pos[0] += self.current_speed_x

                # Calculate tilt angle (in radians)
                tilt_angle = math.atan2(
                    self.current_speed_x, self.current_speed_y)

                # Rotate the drop's pic surface
                rotated_pic = pygame.transform.rotate(
                    self.pic, math.degrees(tilt_angle))

            if self.pos[0] < -51:
                self._reset_on_sides(left=False)
            if self.pos[0] > self.screen_w + 15:
                self._reset_on_sides(left=True)

            # Update the drop's position
            self.pos[1] += self.current_speed_y

            newrect = pygame.Rect(
                self.pos[0], self.pos[1], self.size[0], self.size[1])
            rect = oldrect.union(newrect)

            # Draw the rotated drop
            screen.blit(rotated_pic, self.pos)

            self.current_speed_y += self.acceleration

            if self.pos[1] > self.screen_h:
                self._reset_on_top(wind_speed)

            return rect


class Rain(Precipitation):
    def __init__(self, weather, screen, height=150, width=10, initial_speed=15, acc=5, color=(150, 200, 255, 200), flake=False, num_drops=25, sound_effect_name='rain'):
        super().__init__(weather, screen, height=height, width=width,
                         initial_speed=initial_speed, acc=acc, color=color, flake=flake, num_drops=num_drops)

        if sound_effect_name in ('rain', 'acid rain'):
            weather._ensure_rain_audio_started()
        else:
            for i in range(4):
                sound = pygame.mixer.Sound(f'assets/weather/rain/{i+1}.mp3')
                weather.queue_effect_sound(sound_effect_name, sound)


class Snow(Precipitation):
    """Represents snowfall precipitation.

    Snowflakes can be drawn as circles or squares depending on the pixel parameter.

    Args:
        weather (Weather): Instance of the Weather class.
        screen (pygame.Surface): The Pygame screen where the snow will be drawn.
        height (int): Max height of each snowflake sprite.
        width (int): Max width of each snowflake sprite.
        initial_speed (float): Max initial fall speed for each snowflake.
        acc (float): Max downward acceleration for each snowflake.
        color (tuple[int, int, int, int]): RGBA color of the snowflakes.
        flake (bool): If True, a flake shape is drawn at the bottom of the sprite.
        num_drops (int): Number of snowflakes to maintain on screen.
        pixel (bool): If True, snowflakes are drawn as squares instead of circles.
    """

    def __init__(self, weather, screen: pygame.Surface, height: int = 50, width: int = 25, initial_speed: float = 2,
                 acc: float = 0.1, color: tuple[int, int, int, int] = (255, 255, 255, 255), flake: bool = True,
                 num_drops: int = 35, pixel: bool = False):
        super().__init__(weather=weather, screen=screen, height=height, width=width, initial_speed=initial_speed, acc=acc, color=color,
                         flake=flake, num_drops=num_drops, pixel=pixel)

    def render(self, screen: pygame.Surface, now: float, wind_speed: float) -> pygame.Rect | None:
        """Render the snowflakes, applying amplified wind drift.

        Args:
            screen (pygame.Surface): The Pygame screen where the snowflakes will be drawn.
            now (float): Current time in seconds.
            wind_speed (float): Current wind speed (amplified 4x for snowflakes).

        Returns:
            rect (pygame.Rect): The bounding rectangle where the snowflakes were drawn.
        """
        rect = super().render(screen, now, wind_speed * 4)
        return rect


class Hail(Precipitation):
    """Represents hail precipitation.

    Hailstones bounce when they reach the bottom of the screen and can be drawn as
    circles or squares.

    Args:
        weather (Weather): Instance of the Weather class.
        screen (pygame.Surface): The Pygame screen where the hail will be drawn.
        width (int): Max width of each hailstone sprite.
        height (int): Max height of each hailstone sprite.
        initial_speed (float): Max initial fall speed for each hailstone.
        acc (float): Max downward acceleration for each hailstone.
        color (tuple[int, int, int, int]): RGBA color of the hailstones.
        flake (bool): If True, a flake shape is drawn at the bottom of the sprite.
        num_drops (int): Number of hailstones to maintain on screen.
        pixel (bool): If True, hailstones are drawn as squares instead of circles.
    """

    def __init__(self, weather, screen: pygame.Surface, width: int = 10, height: int = 50, initial_speed: float = 20,
                 acc: float = 1, color: tuple[int, int, int, int] = (200, 200, 200, 255), flake: bool = True,
                 num_drops: int = 10, pixel: bool = False, is_hail: bool = True):
        super().__init__(weather=weather, screen=screen, height=height, width=width, initial_speed=initial_speed, acc=acc, color=color,
                         flake=flake, num_drops=num_drops, pixel=pixel, is_hail=is_hail)
        for i in range(3):
            sound = pygame.mixer.Sound(f'assets/weather/hail/{i+1}.mp3')
            weather.queue_effect_sound('hail', sound)

    class Drop(Precipitation.Drop):
        """A single hailstone drop, with the ability to bounce."""

        def __init__(self, speed: float, acc: float, weight: float, pic: pygame.Surface, screen: pygame.Surface):
            """Initialize a hailstone drop.

            Args:
                speed (float): Initial fall speed of the hailstone.
                acc (float): Downward acceleration per frame.
                weight (float): Mass factor from 0 to 1; higher values resist wind deflection.
                pic (pygame.Surface): Pre-rendered surface representing the hailstone.
                screen (pygame.Surface): The Pygame screen where the hailstone will be drawn.
            """
            super().__init__(speed, acc, weight, pic, screen)
            self.bounce_count = 0  # Track the number of bounces

        def render(self, screen: pygame.Surface, wind_speed: float) -> pygame.Rect | None:
            """Render the hailstone, bouncing it when it hits the bottom of the screen.

            Args:
                screen (pygame.Surface): The Pygame screen where the hailstone will be drawn.
                wind_speed (float): Current wind speed affecting horizontal drift.

            Returns:
                rect (pygame.Rect): The bounding rectangle where the hailstone was drawn.
            """
            rect = super().render(screen, wind_speed)

            if self.pos[1] >= (self.screen_h - 20*(1-self.weight)):
                if self.bounce_count < 5:  # Limit the number of bounces
                    self.current_speed_y = -self.current_speed_y * 0.1  # Lose speed on bounce
                    self.current_speed_x += random.randint(-5, 5)
                    self.bounce_count += 1
                else:
                    self._reset_on_top(wind_speed)
                    self.current_speed_x += random.randint(-5, 5)
                    self.bounce_count = 0

            return rect


class Lightning:
    """Represents lightning flash effects with accompanying thunder sounds.

    Args:
        weather (Weather): Instance of the Weather class.
        screen (pygame.Surface): The Pygame screen where the lightning will be drawn.
        frequency (int): Milliseconds between lightning strikes.
    """

    def __init__(self, weather, screen: pygame.Surface, frequency: int = 8_000):
        self.screen = screen
        self.frequency = frequency
        self.time_for_lightning = random.randint(
            self.frequency - self.frequency // 3, self.frequency + self.frequency // 3)

        self.surface = pygame.Surface(self.screen.get_size())
        self.surface.fill((255, 255, 255))

        self.last_flash_time = pygame.time.get_ticks()
        self.flash_active = False
        self.flash_step = 0
        self.step_duration = []

        self.general_vol = weather.general_vol
        self.effect_vol = weather._audio_scale('lightning')
        self.th_sounds = []
        for i in range(5):
            sound = pygame.mixer.Sound(f'assets/weather/thunders/{i+1}.mp3')
            self.th_sounds.append(sound)
            weather.register_effect_sound('lightning', sound)

    def update(self, wind_speed: float = 0) -> None:
        """
        Update the lightning effect, checking if a flash should start or continue.
        """
        current_time = pygame.time.get_ticks()
        time_since_last_flash = current_time - self.last_flash_time

        if self.flash_active:  # Continue the flash if it is active
            self._continue_flash(current_time)
        elif time_since_last_flash > self.time_for_lightning:  # Checks if a flash should start
            self._start_flash()

        # print(self.step_duration, self.flash_step)

    def _start_flash(self) -> None:
        """Start a lightning flash event."""
        self.time_for_lightning = random.randint(
            self.frequency - self.frequency // 3, self.frequency + self.frequency // 3)
        self.flash_active = True
        self.flash_step = 0
        # The number of flashes in this event.
        number_of_flashes = random.randint(1, 3)
        self.step_duration = [random.randint(200, 300) for _ in range(
            number_of_flashes * 2)]  # Create random durations for the flashes
        if number_of_flashes == 1:
            extra_duration_for_sounds = [random.randint(
                500, 1500)-sum(self.step_duration), 1, 1, 1, 1, 1]
        else:
            extra_duration_for_sounds = [random.randint(
                500, 1500)-sum(self.step_duration), 1, 500, 1, 500, 1]
        # Adds extra time at the end for playing the sounds
        self.step_duration.extend(extra_duration_for_sounds)
        self.flash_step_total = len(self.step_duration)

    def _continue_flash(self, current_time: int) -> None:
        """Continue a lightning flash event."""
        if self.flash_step < self.flash_step_total:
            elapsed_time = current_time - self.last_flash_time

            # First blit the surface. It creates a flashing effect by changing the transparency (alpha) of the surface
            if self.flash_step < self.flash_step_total-6:
                if self.flash_step % 2 == 0:
                    alpha = 70 * \
                        (elapsed_time/self.step_duration[self.flash_step])
                    self.surface.set_alpha(alpha)
                else:
                    alpha = 70 * \
                        (self.step_duration[self.flash_step]/elapsed_time)
                    self.surface.set_alpha(alpha)
                self.screen.blit(self.surface, (0, 0))

            else:  # On the last 6 steps of self.flash_step_total
                if self.flash_step % 2 == 1:
                    chanel = pygame.mixer.find_channel()
                    chanel.set_volume(self.general_vol * self.effect_vol)
                    chanel.play(
                        self.th_sounds[random.randint(0, len(self.th_sounds)-1)])

            # If the time on this step is over, it will jump to the next one
            if elapsed_time >= self.step_duration[self.flash_step]:
                self.flash_step += 1
                self.last_flash_time = current_time

        # On the last step flash_active is swithed off
        elif self.flash_step >= self.flash_step_total:
            self.flash_active = False


class Fog:
    """
    Class representing fog effects.

    :param screen: The Pygame screen where the fog will be drawn.
    :param color: The color of the fog in RGB.
    :param density: A value between 0 and 1 representing how dense the fog is.
    :param pixel: Bollean. If True, it will load a pixelated image.
    :param inertia: From 0 to 100. Being 0 means that the fog image almost follows wind speed, and 100 means that the wind barely affects the movement.
    """

    def __init__(self, screen: pygame.Surface, density: float = 0.7, pixel: bool = False, drag=0, color: tuple[int, int, int] = (20, 0, 20), inertia: int = 10):
        self.screen = screen
        self.screen_w = screen.get_width()
        self.screen_h = screen.get_height()
        self.color = color
        self.inertia = inertia
        self.speed = 0  # Displacement of the image in x direction
        # Amplitude of the sin function that controls the displacement of the image in y direction
        self.y_ampl = random.random()*(self.screen_h // 4)

        self.load_images(density, color, pixel)

    def load_images(self, density, color, pixel):
        # Load images
        if pixel:
            # Available at https://danialc0.itch.io/tileable-fog
            img = pygame.image.load(
                f'assets/weather/NoisePix.png').convert_alpha()
        else:
            img = pygame.image.load(
                f'assets/weather/NoiseReg.png').convert_alpha()
        img = pygame.transform.scale(
            img, (self.screen_w*1.5 - 1, self.screen_h * 2))
        img.set_alpha(int(255*density))

        color_surface = pygame.Surface(img.get_size())
        color_surface.fill(color)
        color_surface.set_alpha(100*density)
        img.blit(color_surface, (0, 0))

        self.img = []
        num_img = 2

        for _ in range(num_img):
            self.img.append(img)

        # Position to start moving the fog from
        self.offset_1 = -self.screen_w*1.5
        self.offset_2 = 1

    def update(self, wind_speed: float) -> None:
        """Update the fog effect, drifting it across the screen based on wind speed.

        Args:
            wind_speed (float): Current wind speed affecting the fog's horizontal drift.
        """
        dx = (wind_speed - self.speed)
        self.speed += dx/(10*self.inertia)

        # Move the fog across the screen in x direction
        self.offset_1 += self.speed

        # Move the fog across the screen in y direction
        y_movement = self.y_ampl * math.sin(pygame.time.get_ticks()/10_000)
        y_pos = -(self.screen_h // 4) + y_movement

        if -0.01 < y_pos < 0.01:
            self.y_ampl = random.random()*(self.screen_h // 4)

        if self.offset_1 > self.screen_w*1.5:
            self.offset_1 = -self.screen_w*1.5
        elif self.offset_1 < -self.screen_w*1.5:
            self.offset_1 = self.screen_w*1.5

        self.offset_2 = self.offset_1 + self.screen_w * \
            1.5 if self.offset_1 <= 0 else self.offset_1 - self.screen_w*1.5

        # Blit the fog surface onto the screen
        self.screen.blit(self.img[0], (self.offset_1, y_pos))
        self.screen.blit(self.img[1], (self.offset_2, y_pos))


def stream_agent_response(user_prompt: str, weather_context: str = "", previous_narrative: str = ""):
    """Stream the narrator's text response token by token.

    Args:
        user_prompt (str): The raw user request to narrate.
        weather_context (str): JSON string of the weather values already applied
            to the simulator so the narrative can reference the actual state.
        previous_narrative (str): Most recent narrator text for scene continuity.

    Yields:
        chunk (str): A streamed text chunk from the language model.
    """
    yield from generate_weather_narrative(user_prompt, weather_context, previous_narrative)


def generate_weather_state(user_prompt: str, narrative: str = ""):
    """Return a WeatherState from the JSON model.

    Args:
        user_prompt (str): The raw user request.
        narrative (str): Optional narrator text for extra context. Pass "" to
            generate weather values before the narrative exists.

    Returns:
        state (WeatherState): Validated weather state, or None on error.
    """
    return generate_json_output(user_prompt, narrative)


def wrap_text_for_width(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Wrap text using actual font pixel width instead of only character count."""
    if not text:
        return [""]

    wrapped_lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    wrapped_lines.append(current)
                current = word

                # Hard-wrap very long tokens such as URLs or pasted JSON
                while font.size(current)[0] > max_width and len(current) > 1:
                    cut = max(1, int(len(current) * max_width /
                              max(1, font.size(current)[0])))
                    wrapped_lines.append(current[:cut])
                    current = current[cut:]

        wrapped_lines.append(current)

    return wrapped_lines


class ChatSystem:
    """Bottom-screen chat UI with streamed agent responses."""

    def __init__(self, rect: pygame.Rect, font: pygame.font.Font, small_font: pygame.font.Font,
                 weather, sliders: list, app_state: dict):
        self.rect = rect
        self.font = font
        self.small_font = small_font
        self.weather = weather
        self.sliders = sliders
        self.app_state = app_state
        self.messages: list[tuple[str, str]] = []
        self.input_text = ""
        self.input_all_selected = False
        self.active = False
        self.streaming = False
        self.stream_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        # Procedural evolution: fire every evolution_interval seconds when idle
        self.evolution_interval: float = 20.0
        self.last_evolution_time: float = time.time()
        # Background image generation state
        self._image_queue: queue.Queue = queue.Queue()
        self._pending_image: bool = False
        self._current_struct: str | None = None
        # Slider debounce: wait for the user to stop moving sliders before
        # firing a narrative, and push the evolution clock back.
        self._slider_debounce_deadline: float = 0.0
        self._pending_slider_values: dict = {}

    def _command_modifier_down(self) -> bool:
        mods = pygame.key.get_mods()
        mac_command = getattr(pygame, "KMOD_META", 0) | getattr(
            pygame, "KMOD_GUI", 0)
        return bool(mods & (pygame.KMOD_CTRL | mac_command))

    # Clipboard helpers for Ctrl/Cmd+C, Ctrl/Cmd+X, and Ctrl/Cmd+V
    def _copy_text_to_clipboard(self, text: str) -> None:
        if not text:
            return
        try:
            pygame.scrap.init()
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        except Exception:
            # Clipboard support varies by platform/display backend
            pass

    def _paste_text_from_clipboard(self) -> str:
        try:
            pygame.scrap.init()
            pasted = pygame.scrap.get(pygame.SCRAP_TEXT)
            if pasted:
                return pasted.decode("utf-8", errors="ignore").replace("\x00", "")
        except Exception:
            pass
        return ""

    # Replace selected input text when typing/pasting like a normal field
    def _insert_text(self, text: str) -> None:
        if not text:
            return
        if self.input_all_selected:
            self.input_text = text
            self.input_all_selected = False
        else:
            self.input_text += text

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True when the event was consumed by the chat box."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
            if not self.active:
                self.input_all_selected = False
            return self.active

        if event.type != pygame.KEYDOWN or not self.active:
            return False

        command_down = self._command_modifier_down()

        if event.key == pygame.K_RETURN:
            self.input_all_selected = False
            self.submit_prompt()

        elif event.key == pygame.K_BACKSPACE:
            if self.input_all_selected:
                # Backspace deletes selected text
                self.input_text = ""
                self.input_all_selected = False
            elif command_down:
                # Ctrl/Cmd + Backspace clears the whole current prompt
                self.input_text = ""
            else:
                # Normal Backspace deletes one character
                self.input_text = self.input_text[:-1]

        elif event.key == pygame.K_DELETE:
            if self.input_all_selected:
                # Delete also removes selected text
                self.input_text = ""
                self.input_all_selected = False

        elif event.key == pygame.K_ESCAPE:
            # Escape clears selection first, then text, then focus
            if self.input_all_selected:
                self.input_all_selected = False
            elif self.input_text:
                self.input_text = ""
            else:
                self.active = False

        elif event.key == pygame.K_a and command_down:
            # Ctrl/Cmd + A selects the whole prompt
            self.input_all_selected = bool(self.input_text)

        elif event.key == pygame.K_c and command_down:
            # Ctrl/Cmd + C copies selected input text
            if self.input_all_selected:
                self._copy_text_to_clipboard(self.input_text)

        elif event.key == pygame.K_x and command_down:
            # Ctrl/Cmd + X cuts selected input text
            if self.input_all_selected:
                self._copy_text_to_clipboard(self.input_text)
                self.input_text = ""
                self.input_all_selected = False
        # TODO: Bruh for some reason this one doesnt't work (at least on
        # MacOS)
        elif event.key == pygame.K_v and command_down:
            # Ctrl/Cmd + V pastes and replaces selected text if all is selected
            pasted = self._paste_text_from_clipboard()
            self._insert_text(pasted)

        elif event.key == pygame.K_u and command_down:
            # Terminal-style shortcut. Ctrl/Cmd + U clears the current prompt
            self.input_text = ""
            self.input_all_selected = False

        elif event.unicode and event.unicode.isprintable():
            self._insert_text(event.unicode)

        return True

    def submit_prompt(self) -> None:
        prompt = self.input_text.strip()
        if not prompt or self.streaming:
            return

        self.input_text = ""
        self._current_struct = None
        self.messages.append(("You", prompt))
        self.messages.append(("Agent", ""))
        self.streaming = True

        def worker() -> None:
            # Generate structured weather values from the user prompt
            # and apply them to the sliders before the narrative begins
            payload = {}
            try:
                weather_state = generate_weather_state(prompt)

                # Accept either a Pydantic model or a plain dict from ai_backend
                if hasattr(weather_state, "model_dump"):
                    payload = weather_state.model_dump()
                elif isinstance(weather_state, dict):
                    payload = weather_state
                else:
                    raise TypeError(
                        f"generate_weather_state returned {type(weather_state).__name__}, "
                        "expected Pydantic model or dict"
                    )

                self.stream_queue.put(("weather_json", payload))
            except Exception as e:
                self.stream_queue.put(
                    ("system", f"Error generating weather JSON: {e}"))
                self.stream_queue.put(("done", ""))
                return

            # Stream the narrative using both the original prompt and
            # the structured weather values so it matches the simulation state
            try:
                weather_context = json.dumps(
                    payload, indent=2) if payload else ""
                narrative_received = False
                for chunk in stream_agent_response(prompt, weather_context):
                    if chunk:
                        narrative_received = True
                    self.stream_queue.put(("delta", chunk))
                if not narrative_received:
                    self.stream_queue.put((
                        "system",
                        "Narrative generation returned no output (possible rate limit). "
                        "Check the console for details."
                    ))
            except Exception as e:
                self.stream_queue.put((
                    "system", f"Error generating narrative: {e}"
                ))

            self.stream_queue.put(("done", ""))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def submit_slider_narrative(self, weather_values: dict) -> None:
        """Stream a short narrative driven entirely by the current slider state.

        Called after the user releases a slider. Skips silently if a stream is already
        in progress.

        Args:
            weather_values (dict): Current weather values from
                Weather.get_weather_effect_values(); used as context for the narrator.
        """
        if self.streaming:
            return

        self._current_struct = None
        last_narrative = next(
            (text for spk, text in reversed(self.messages) if spk == "Agent"),
            ""
        )
        self.messages.append(("Agent", ""))
        self.streaming = True
        weather_context = json.dumps(weather_values, indent=2)

        def worker() -> None:
            try:
                narrative_received = False
                for chunk in stream_agent_response(
                    "Describe the current weather conditions.", weather_context, last_narrative
                ):
                    if chunk:
                        narrative_received = True
                    self.stream_queue.put(("delta", chunk))
                if not narrative_received:
                    self.stream_queue.put((
                        "system",
                        "Narrative generation returned no output (possible rate limit). "
                        "Check the console for details."
                    ))
            except Exception as e:
                self.stream_queue.put(
                    ("system", f"Error generating narrative: {e}"))
            self.stream_queue.put(("done", ""))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def request_background_update(self, weather_values: dict, narrative: str) -> None:
        """Spawn a daemon thread that generates a background image via Pollinations.ai.

        The raw image bytes are placed on _image_queue for the main loop to consume.

        Args:
            weather_values (dict): Current weather state used to build the image prompt.
            narrative (str): Most recent narrator text providing visual context.
        """
        weather_context = json.dumps(weather_values, indent=2)

        def worker() -> None:
            image_prompt = generate_image_prompt(narrative, weather_context)
            if not image_prompt:
                self.stream_queue.put(
                    ("system", "Background image error: image prompt generation returned nothing.")
                )
                return
            print(
                f"Generating background image for prompt: {image_prompt[:80]}...")
            image_bytes = generate_background_image(image_prompt)
            if image_bytes:
                self._image_queue.put(image_bytes)
            else:
                self.stream_queue.put(
                    (
                        "system",
                        "Background image error: image generation failed or returned no image. "
                        "This can happen if the image API is rate-limited."
                    )
                )

        self._pending_image = True
        threading.Thread(target=worker, daemon=True).start()

    def poll_background_image(self, screen_size: tuple) -> "pygame.Surface | None":
        """Check whether a background image has been generated.

        Args:
            screen_size (tuple): (width, height) to scale the loaded image to.

        Returns:
            surface (pygame.Surface): Scaled background surface if one is ready, else None.
        """
        try:
            image_bytes = self._image_queue.get_nowait()
            buf = io.BytesIO(image_bytes)
            surface = pygame.image.load(buf)
            self._pending_image = False
            return pygame.transform.scale(surface, screen_size).convert()
        except queue.Empty:
            return None
        except Exception as e:
            print(f"Error loading generated background image: {e}")
            self._pending_image = False
            return None

    def notify_slider_activity(self, weather_values: dict) -> None:
        """Buffer the latest slider state and reset both debounce and
        weather evolution timers."""
        self._slider_debounce_deadline = time.time() + 1.5
        self._pending_slider_values = weather_values
        self.last_evolution_time = time.time()

    def tick_slider_debounce(self) -> None:
        """Fire the slider narrative once the debounce window has expired."""
        if not self._pending_slider_values:
            return
        if time.time() < self._slider_debounce_deadline:
            return
        if self.streaming:
            return
        values = self._pending_slider_values
        self._pending_slider_values = {}
        self.submit_slider_narrative(values)

    def tick_evolution(self) -> None:
        """
        Called once per frame. Triggers a weather evolution when the idle
        interval has elapsed and no stream is in progress.
        """
        if self.streaming:
            return
        if time.time() - self.last_evolution_time < self.evolution_interval:
            return
        self.last_evolution_time = time.time()
        self.submit_weather_evolution()

    def submit_weather_evolution(self) -> None:
        """
        Slightly evolve the current weather state and stream a short narrative.

        Calls the JSON model with current slider values as context, applies the
        evolved state to the simulator, then generates a brief narrator update.
        Does not accept any user text input — this path is fully automated.
        Skips silently if a stream is already in progress.
        """
        if self.streaming:
            return

        current_values = self.weather.get_weather_effect_values(
            self.sliders, self.app_state)
        self._current_struct = None
        self.messages.append(("Agent", ""))
        self.streaming = True

        last_narrative = next(
            (text for spk, text in reversed(self.messages) if spk == "Agent"),
            ""
        )

        def worker() -> None:
            # Generate evolved weather values from the current state
            # and narrative so it stays coherent.
            payload = {}
            try:
                evolved = generate_weather_evolution(
                    current_values, last_narrative)
                if evolved is not None:
                    if hasattr(evolved, "model_dump"):
                        payload = evolved.model_dump()
                    elif isinstance(evolved, dict):
                        payload = evolved
                    else:
                        raise TypeError(
                            f"generate_weather_evolution returned "
                            f"{type(evolved).__name__}, expected Pydantic model or dict"
                        )
                    self.stream_queue.put(("weather_json", payload))
            except Exception as e:
                self.stream_queue.put(
                    ("system", f"Weather evolution error: {e}"))

            # Stream a short narrative continuing the established scene
            try:
                weather_context = json.dumps(
                    payload, indent=2) if payload else ""
                narrative_received = False
                for chunk in stream_agent_response(
                    "The weather is gently evolving.", weather_context, last_narrative
                ):
                    if chunk:
                        narrative_received = True
                    self.stream_queue.put(("delta", chunk))
                if not narrative_received:
                    self.stream_queue.put((
                        "system",
                        "Narrative generation returned no output (possible rate limit). "
                        "Check the console for details."
                    ))
            except Exception as e:
                self.stream_queue.put(
                    ("system", f"Error generating narrative: {e}"))

            self.stream_queue.put(("done", ""))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def poll_stream(self) -> None:
        while True:
            try:
                kind, payload = self.stream_queue.get_nowait()
            except queue.Empty:
                break
            # Agent finished generating weather dialogue
            if kind == "delta":
                if self.messages and self.messages[-1][0] == "Agent":
                    speaker, text = self.messages[-1]
                    self.messages[-1] = (speaker, text + payload)
            # Agent finished generating JSON structure from user prompt
            elif kind == "weather_json":
                self._apply_weather_json(payload)
                self._current_struct = json.dumps(payload, indent=2)
            elif kind == "system":
                # Show backend / JSON errors in the chat
                self.messages.append(("System", payload))
            # Agent done performing all actions
            elif kind == "done":
                last_agent_text = next(
                    (text for spk, text in reversed(
                        self.messages) if spk == "Agent"),
                    ""
                )
                print(f"\nSystem generated text output:\n{last_agent_text}\n")
                if self._current_struct:
                    print(
                        f"System applied weather JSON:\n{self._current_struct}\n")

                self.streaming = False
                self.last_evolution_time = time.time()

                # Trigger a background image regeneration when a weather_json
                # was applied this cycle
                if self._current_struct:
                    new_weather = json.loads(self._current_struct)

                    narrative = next(
                        (
                            text for spk, text in reversed(self.messages)
                            if spk == "Agent"
                        ),
                        ""
                    )

                    self.request_background_update(new_weather, narrative)

    def _apply_weather_json(self, data: dict) -> None:
        """Apply validated JSON model output to the simulator."""
        # Create weather mappings based on JSON model structure
        mapping = {
            "rain": data.get("rain_intensity", data.get("rain", 0.0)),
            "acid rain": data.get("acid_rain_intensity", data.get("acid_rain", 0.0)),
            "snow": data.get("snow_intensity", data.get("snow", 0.0)),
            "hail": data.get("hail_intensity", data.get("hail", 0.0)),
            "fog": data.get("fog_density", data.get("fog", 0.0)),
            "wind": data.get("wind_speed", data.get("wind", 0.0)),
            "lightning": data.get("lightning_intensity", data.get("lightning_frequency", data.get("lightning", 0.0))),
            "volume": data.get("volume", data.get("volume_level", 1.0)),
        }

        # Set each slider to their respective value
        for slider in self.sliders:
            if slider.effect_name in mapping:
                slider.value = max(
                    0.0, min(1.0, float(mapping[slider.effect_name])))
                self.weather.set_effect_intensity(
                    slider.effect_name, slider.value)
        self.app_state["brightness"] = max(
            0.0, min(1.0, float(data.get("brightness", 1.0))))
        self.app_state["cloud_cover"] = max(
            0.0, min(1.0, float(data.get("cloud_cover", 0.0))))
        self.app_state["visibility"] = max(
            0.0, min(1.0, float(data.get("visibility", 1.0))))

    def draw(self, screen: pygame.Surface) -> None:
        self.poll_stream()

        panel = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel.fill((8, 10, 16, 210))
        screen.blit(panel, self.rect.topleft)
        pygame.draw.rect(screen, (110, 130, 155),
                         self.rect, 1, border_radius=8)

        title = self.font.render(
            "Click the box, type a prompt, and press enter to send the weather simulator agent a prompt:", True, (245, 245, 245))
        screen.blit(title, (self.rect.x + 16, self.rect.y + 10))

        input_height = 34
        input_rect = pygame.Rect(
            self.rect.x + 16,
            self.rect.bottom - input_height - 12,
            self.rect.width - 32,
            input_height,
        )
        messages_rect = pygame.Rect(
            self.rect.x + 16,
            self.rect.y + 38,
            self.rect.width - 32,
            input_rect.y - (self.rect.y + 44),
        )

        # Draw newest lines that fit in the message area
        rendered_lines: list[tuple[str, tuple[int, int, int]]] = []
        for speaker, text in self.messages:
            if speaker == "You":
                color = (220, 235, 210)
            elif speaker == "System":
                color = (255, 150, 150)
            else:
                color = (120, 255, 150)

            body = text
            if speaker == "Agent" and self.streaming and (speaker, text) == self.messages[-1]:
                body += "▌"
            prefix = f"{speaker}: "
            wrapped = wrap_text_for_width(
                prefix + body, self.small_font, messages_rect.width)
            for line in wrapped:
                rendered_lines.append((line, color))
            rendered_lines.append(("", color))

        line_height = self.small_font.get_height() + 3
        max_lines = max(1, messages_rect.height // line_height)
        visible_lines = rendered_lines[-max_lines:]

        y = messages_rect.y
        for line, color in visible_lines:
            if line:
                text_surface = self.small_font.render(line, True, color)
                screen.blit(text_surface, (messages_rect.x, y))
            y += line_height

        input_color = (32, 38, 48) if self.active else (22, 26, 34)
        border_color = (170, 210, 255) if self.active else (95, 110, 130)
        pygame.draw.rect(screen, input_color, input_rect, border_radius=6)
        pygame.draw.rect(screen, border_color, input_rect, 2, border_radius=6)

        # Draw selected input with a highlight when Ctrl/Cmd+A is used
        placeholder = "Ask the weather simulator agent..." if not self.input_text else self.input_text
        cursor = "|" if self.active and (
            pygame.time.get_ticks() // 500) % 2 == 0 and not self.input_all_selected else ""

        if self.input_text and self.input_all_selected:
            selected_text = self.input_text
            selected_surface = self.font.render(
                selected_text, True, (255, 255, 255))
            highlight_rect = pygame.Rect(
                input_rect.x + 8,
                input_rect.y + 5,
                min(selected_surface.get_width() + 6, input_rect.width - 16),
                input_rect.height - 10,
            )
            pygame.draw.rect(screen, (70, 120, 190),
                             highlight_rect, border_radius=4)
            screen.blit(selected_surface,
                        (input_rect.x + 10, input_rect.y + 8))
        else:
            color = (135, 145, 160) if not self.input_text else (245, 245, 245)
            input_surface = self.font.render(placeholder + cursor, True, color)
            screen.blit(input_surface, (input_rect.x + 10, input_rect.y + 8))


class Slider:
    """Simple pygame slider for normalized values between 0.0 and 1.0."""

    def __init__(self, label: str, effect_name: str, x: int, y: int, width: int, initial: float = 0.0):
        self.label = label
        self.effect_name = effect_name
        self.x = x
        self.y = y
        self.width = width
        self.height = 22
        self.value = max(0.0, min(1.0, float(initial)))
        self.dragging = False
        self.just_released = False  # True for one event cycle after a drag ends
        self.knob_radius = 9

    def _set_from_mouse_x(self, mouse_x: int) -> None:
        self.value = max(0.0, min(1.0, (mouse_x - self.x) / self.width))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True when the slider value changed. Sets just_released=True on drag end."""
        self.just_released = False
        changed = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            knob_x = self.x + self.value * self.width
            knob_hit = (mouse_x - knob_x) ** 2 + (mouse_y -
                                                  self.y) ** 2 <= (self.knob_radius + 4) ** 2
            bar_hit = self.x <= mouse_x <= self.x + \
                self.width and self.y - 8 <= mouse_y <= self.y + 8

            if knob_hit or bar_hit:
                self.dragging = True
                self._set_from_mouse_x(mouse_x)
                changed = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.just_released = True
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_mouse_x(event.pos[0])
            changed = True

        return changed

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        label_surface = font.render(
            f"{self.label}: {int(self.value * 100):3d}%", True, (245, 245, 245))
        screen.blit(label_surface, (self.x, self.y - 24))

        pygame.draw.line(screen, (90, 90, 90), (self.x, self.y),
                         (self.x + self.width, self.y), 6)
        pygame.draw.line(screen, (170, 210, 255), (self.x, self.y),
                         (self.x + int(self.value * self.width), self.y), 6)

        knob_x = int(self.x + self.value * self.width)
        pygame.draw.circle(screen, (245, 245, 245),
                           (knob_x, self.y), self.knob_radius)
        pygame.draw.circle(screen, (35, 35, 35),
                           (knob_x, self.y), self.knob_radius, 2)


def draw_slider_panel(screen: pygame.Surface, sliders: list[Slider], font: pygame.font.Font, panel_x: int, panel_y: int) -> None:
    """Draw the translucent control panel after weather effects render."""
    panel_width = 360
    panel_height = 52 + len(sliders) * 46
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((10, 10, 15, 185))
    screen.blit(panel, (panel_x, panel_y))
    pygame.draw.rect(screen, (100, 120, 140), (panel_x, panel_y,
                     panel_width, panel_height), 1, border_radius=8)

    title = font.render("Weather Controls", True, (255, 255, 255))
    screen.blit(title, (panel_x + 16, panel_y + 10))

    for slider in sliders:
        slider.draw(screen, font)


def main():
    SCREENSIZE = 1200, 800
    PIXEL = True

    pygame.init()
    screen = pygame.display.set_mode(SCREENSIZE)
    pygame.display.set_caption(
        "Real-Time Generative Weather Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    chat_font = pygame.font.SysFont(None, 20)

    # Weather options: ['rain', 'acid rain', 'snow', 'hail', 'lightning', 'fog']
    weather = Weather(screen, weather_types=[], wind_speed=0, pixel=PIXEL)
    weather.change_volume(1.0)

    slider_panel_width = 360
    slider_panel_x = SCREENSIZE[0] - slider_panel_width
    slider_panel_y = 16
    slider_x = slider_panel_x + 16
    slider_width = 280

    sliders = [
        Slider("Rain", "rain", slider_x, 100, slider_width, 0.0),
        Slider("Acid rain", "acid rain", slider_x, 146, slider_width, 0.0),
        Slider("Snow", "snow", slider_x, 192, slider_width, 0.0),
        Slider("Hail", "hail", slider_x, 238, slider_width, 0.0),
        Slider("Lightning", "lightning", slider_x, 284, slider_width, 0.0),
        Slider("Fog", "fog", slider_x, 330, slider_width, 0.0),
        Slider("Wind", "wind", slider_x, 376, slider_width, 0.0),
        Slider("Volume", "volume", slider_x, 422, slider_width, 1.0),
    ]

    for slider in sliders:
        weather.set_effect_intensity(slider.effect_name, slider.value)

    app_state = {
        "brightness": 1.0,
        "cloud_cover": 0.0,
        "visibility": 1.0,
    }

    # Interactable chat box that displays user prompts and streams agent responses
    # token-by-token/chunk-by-chunk while simulator keeps running
    chat_rect = pygame.Rect(
        16, SCREENSIZE[1] - 210, SCREENSIZE[0] - 32, 208)
    chat_sys = ChatSystem(chat_rect, font, chat_font,
                          weather, sliders, app_state)

    background = pygame.image.load(f'assets/weather/imgpix.webp').convert_alpha(
    ) if PIXEL else pygame.image.load(f'assets/weather/img.webp').convert_alpha()
    background = pygame.transform.scale(
        background, (SCREENSIZE[0], SCREENSIZE[1]))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            # User submitted a prompt in the chat box
            if chat_sys.handle_event(event):
                continue

            # Slider values were changed
            user_modified_slider = False
            slider_just_released = False
            for slider in sliders:
                if slider.handle_event(event):
                    user_modified_slider = True
                    weather.set_effect_intensity(
                        slider.effect_name, slider.value)
                if slider.just_released:
                    slider_just_released = True

            # Buffer slider release; debounce fires the narrative after inactivity
            if slider_just_released:
                weather_values = weather.get_weather_effect_values(
                    sliders, app_state)
                chat_sys.notify_slider_activity(weather_values)

        # Swap in a newly generated background if one arrived from Pollinations.ai
        new_bg = chat_sys.poll_background_image(SCREENSIZE)
        if new_bg is not None:
            background = new_bg

        screen.blit(background, (0, 0))
        weather.update()
        chat_sys.tick_slider_debounce()
        chat_sys.tick_evolution()

        # Draw UI last so both the right slider menu and bottom chat stay readable
        draw_slider_panel(screen, sliders, font,
                          slider_panel_x, slider_panel_y)
        chat_sys.draw(screen)

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()
