# Real-Time Generative Weather Simulator

A real-time generative weather simulation built with **pygame**, **LangChain**, **Groq-hosted LLMs**, and **Pollination.AI** image generation.

This project lets users describe a weather scene in natural language, then generates and simulates that environment interactively. Users can type prompts such as:

> A hot, sunny day on a beach

The application uses language models to interpret the prompt, convert it into structured weather parameters, and update the simulation in real time. Users can also manually adjust weather sliders to fine-tune the scene.

## Credits for Baseline Simulation:

Author: Dani Alc

Mail: danialcdev@gmail.com

Original GitHub Repository link: https://github.com/DaniAlc0/weather

This code may be used and/or modified in any non-commerical project.

## Features

* Real-time weather simulation using `pygame`
* Natural-language prompt input for generating weather scenes
* Interactive weather controls and sliders
* Text-to-text generation with `llama-3.3-70b-versatile`
* Text-to-JSON generation using `openai/gpt-oss-20b` with Pydantic validation
* AI-generated scene imagery through `Pollination.AI`
* LangChain-based orchestration
* `Groq` API integration for low latency LLM inference

## Tech Stack

* **Python 3**
* **pygame**
* **LangChain**
* **Groq**
* **Pydantic**
* **llama-3.3-70b-versatile**
* **openai/gpt-oss-20b**
* **Pollination.AI**

## Installation

### 0. Prerequisites

Before creating the virtual environment, make sure you have Python 3.9–3.13 installed.

Check your Python version with:

```bash
python --version
```
or:
```bash
python3 --version
```

If Python is not installed, download and install a compatible version from the official Python website before continuing: https://www.python.org/downloads/

### 1. Clone the GitHub repository

Clone the project from GitHub and move into the project directory:

```bash
git clone https://github.com/jreaume5/generative-weather-sim.git
cd generative-weather-sim
```

### 2. Create a virtual environment

Make sure any Python 3.9-3.13 is installed before creating the virtual environment.

```bash
python -m venv .venv
```

Activate the virtual environment:

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 3. Obtain a Groq API key

Create a free account or log in at:

```text
https://console.groq.com
```

Then generate an API key from the Groq dashboard.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set your environment variable

#### Linux / macOS

```bash
export GROQ_API_KEY="your_groq_key_here"
```

#### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_groq_key_here"
```

## Running the Application

Start the simulator with:

```bash
python weather.py
```
or
```bash
python3 weather.py
```


## Using the Simulator

After running the application, a `pygame` window will open displaying the **Real-Time Generative Weather Simulator**.

You can interact with the simulator in two main ways:

1. Click the chat box and enter a natural-language weather prompt.
2. Manually adjust the weather sliders to control the simulation parameters.

Example prompts:

```text
A hot, sunny day on a beach
```

```text
A foggy morning in a quiet forest
```

```text
A thunderstorm rolling over a futuristic city
```

The system will interpret the prompt, generate structured weather settings, and update the visual simulation accordingly.

## Notes

Image generation may take 10–15 seconds or longer depending on network conditions, API response time, and system performance.

Make sure your Groq API key is set correctly before launching the application. If the key is missing or invalid, the text-generation components will not work properly, which will impact the quality of other generated outputs.

## Requirements

* Python 3.9-3.13
* Groq API key
* Internet connection for API-based text and image generation
