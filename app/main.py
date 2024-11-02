from fastapi import FastAPI
import neural_net
import time
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class Vector(BaseModel):
    vector: List[float]

@app.post("/mlp")
async def neunet(nodos_ocultos: int,
                 salidas: int, 
                 muestra: Vector):
    start = time.time()
    
    size = len(muestra.vector)

    # Crear una red neuronal con n entradas, n nodos en la capa oculta y n salidas
    nn = neural_net.NeuralNetwork(size, nodos_ocultos, salidas)

    # Propagación hacia adelante con datos de entrada
    result = nn.forward(muestra.vector)

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Resultado": result,
        "muestra": muestra.vector
    }
    jj = json.dumps(j1)

    return jj