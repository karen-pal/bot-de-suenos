from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware  # Importa CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

from fastapi.responses import FileResponse
import os

# Inicializa FastAPI
app = FastAPI()

# Permite solicitudes de cualquier origen (ajusta esto según tus necesidades)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes; cambia "*" por la URL de tu frontend si quieres limitar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todos los encabezados
)

# Carga el tokenizador y el modelo al arrancar
tokenizer = AutoTokenizer.from_pretrained("kar-pal/botu")
model = AutoModelForCausalLM.from_pretrained("kar-pal/botu", device_map="auto", trust_remote_code=True)
public_url = None

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. You are BotU, a dreams bot. You do not answer any question outside of questions about dreams and nightmares. When asked out of domain questions, you answer with a dream. You only answer in spanish.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Modelo de datos para FastAPI
class PromptRequest(BaseModel):
    instruction: str
    user_input: str
# Función para generar una imagen PNG
def generar_imagen(texto: str, archivo_salida: str = "respuesta.png"):
    # Dimensiones y configuraciones de la imagen
    ancho, alto = 1920, 1080
    fondo_color = "black"
    texto_color = "white"

    # Crear imagen negra
    imagen = Image.new("RGB", (ancho, alto), color=fondo_color)
    draw = ImageDraw.Draw(imagen)

    # Cargar fuente (usa una fuente del sistema o una incluida)
    try:
        fuente = ImageFont.truetype("ARIAL.TTF", size=120)
    except IOError:
        print("didnt find font")
        fuente = ImageFont.load_default()

    # Calcular posición del texto para centrarlo
    bbox = draw.textbbox((0, 0), texto, font=fuente)
    texto_ancho, texto_alto = bbox[2] - bbox[0], bbox[3] - bbox[1]
    posicion = ((ancho - texto_ancho) // 2, (alto - texto_alto) // 2)

    # Dibujar el texto
    draw.text(posicion, texto, fill=texto_color, font=fuente)

    # Guardar la imagen
    print("saved image")
    imagen.save(archivo_salida)
def generar_imagen_responsive(texto: str, archivo_salida: str = "respuesta.png"):
    # Dimensiones de la imagen
    ancho, alto = 1920, 1080
    fondo_color = "black"
    texto_color = "white"

    # Crear imagen negra
    imagen = Image.new("RGB", (ancho, alto), color=fondo_color)
    draw = ImageDraw.Draw(imagen)

    # Cargar fuente y ajustar tamaño dinámicamente
    try:
        fuente_path = "ARIAL.TTF"  # Cambiar por la ruta de una fuente válida en tu sistema
        tamaño_fuente = 120  # Tamaño inicial
        fuente = ImageFont.truetype(fuente_path, size=tamaño_fuente)
    except IOError:
        print("Fuente no encontrada, usando la predeterminada.")
        fuente = ImageFont.load_default()
        tamaño_fuente =60  # Valor arbitrario inicial si se usa la fuente por defecto

    # Ajustar el tamaño de la fuente para que el texto quepa en la imagen
    while True:
        bbox = draw.textbbox((0, 0), texto, font=fuente)
        texto_ancho, texto_alto = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if texto_ancho <= ancho * 0.9 and texto_alto <= alto * 0.9:  # Dejar un margen del 10%
            break
        tamaño_fuente -= 2  # Reducir el tamaño de la fuente gradualmente
        if tamaño_fuente < 10:  # Límite mínimo para evitar bucles infinitos
            print("Texto demasiado grande para ajustarlo.")
            break
        fuente = ImageFont.truetype(fuente_path, size=tamaño_fuente)

    # Calcular posición del texto para centrarlo
    bbox = draw.textbbox((0, 0), texto, font=fuente)
    texto_ancho, texto_alto = bbox[2] - bbox[0], bbox[3] - bbox[1]
    posicion = ((ancho - texto_ancho) // 2, (alto - texto_alto) // 2)

    # Dibujar el texto
    draw.text(posicion, texto, fill=texto_color, font=fuente)

    # Guardar la imagen
    imagen.save(archivo_salida)
    print(f"Imagen guardada en {archivo_salida}")
@app.post("/generate/")
async def generate_response(request: PromptRequest):
    #try:
    # Construye el input para el modelo
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                request.instruction, 
                request.user_input,
                ""
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    # Genera la respuesta
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extraer la parte después de "### Response: "
    respuesta_limpia = response.split("### Response:")[-1].strip()
    print(respuesta_limpia)
    archivo_imagen = "respuesta.png"

    # Generar la imagen con la respuesta
    generar_imagen_responsive(respuesta_limpia)
    # Construir la URL pública de la imagen
    if public_url:
        imagen_url = f"{public_url}/{archivo_imagen}"
    else:
        imagen_url = f"http://localhost:8001/{archivo_imagen}"

    print(f"Imagen disponible en: {imagen_url}")

    return {"response": respuesta_limpia, "image_url": imagen_url}
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))
# Endpoint para servir la imagen generada
@app.get("/respuesta.png")
async def get_image():
    archivo_salida = "respuesta.png"
    if os.path.exists(archivo_salida):
        return FileResponse(archivo_salida, media_type="image/png")
    raise HTTPException(status_code=404, detail="Imagen no encontrada")


# Configuración del túnel con pyngrok
if __name__ == "__main__":
    from fastapi.openapi.utils import get_openapi
    from uvicorn import run

    # Inicia ngrok
    public_url = ngrok.connect(8001).public_url
    print(f"Servidor público disponible en: {public_url}")
    print(f"Imagen disponible en: {public_url}/respuesta.png")


    # Ejecuta FastAPI
    run(app, host="0.0.0.0", port=8001)
