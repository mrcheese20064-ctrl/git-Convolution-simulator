# Simulador de convolución

Simulador interactivo de convolución **continua** y **discreta**:

- Continua: `y(t) = ∫ x(τ)·h(t−τ) dτ`
- Discreta: `y[n] = Σ x[k]·h[n−k]`

Disponible en dos versiones: una de escritorio (Python) y una web que corre directo en el navegador.

## Versión web (sin instalar nada)

Abre `docs/index.html` en cualquier navegador, o usa la versión publicada con GitHub Pages:

```
https://mrcheese20064-ctrl.github.io/simulador-convolucion/
```

## Versión de escritorio (Python + PyQt6)

```bash
git clone https://github.com/<tu-usuario>/simulador-convolucion.git
cd simulador-convolucion
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Requiere Python 3.9 o superior.

## Pestañas

1. **Señales**: ingreso de `x(t)` y `h(t)`.
2. **Reflexión**: `h(−τ)` reflejada.
3. **Animación**: desplazamiento de `h(t−τ)` y construcción de `y(t)`.
4. **Resultado**: `y(t)` completo.

Un selector **CONTINUO / DISCRETO** (arriba a la izquierda) cambia el modo.

## Si algo falla

Los errores quedan registrados en `convolucion_error.log` junto a `main.py`. Si abres un issue, adjunta ese archivo.

## Estructura

```
├── main.py            # app de escritorio
├── docs/index.html    # versión web (GitHub Pages)
├── requirements.txt
├── LICENSE
└── README.md
```

## Licencia

MIT. Haz lo que quieras con él, solo conserva el aviso de copyright.
