<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.md">English</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/understanding-dataset-lab/readme.png" alt="understanding-dataset-lab" width="400">
</p>

<p align="center">
  <b>empathy</b>: "Identification with or understanding of the thoughts, feelings, or emotional state of another person."<br>
  <sub>Definition from <a href="https://en.wiktionary.org/wiki/empathy">Wiktionary</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub><br><br>
  Cognitive empathy, as udlab means it: an accurate model of what each party in a situation knows, believes, wants and needs.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-prototype-orange" alt="Prototype">
  <a href="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/understanding-dataset-lab/"><img src="https://img.shields.io/badge/docs-landing_page-blue" alt="Landing page"></a>
</p>

> **Prototipo.** udlab es una investigación abierta en curso. El método y el código están disponibles para leer y ejecutar. Sus resultados
> hasta ahora son comprobaciones de qué tan bien genera escenas, en bancos de desarrollo. No tiene un resultado final o definitivo, ni
> un modelo entrenado, y no hace ninguna afirmación sobre si alguna IA comprende a alguien. Las etapas aún pueden cambiar la forma en que se crean las escenas.

**understanding-dataset-lab (udlab) crea escenarios de prueba para la empatía cognitiva: si un modelo puede mantener una
imagen precisa de lo que cada parte en una escena sabe, cree, desea y necesita.** Se planifican escenas cortas con varias personas
mediante código, y luego se redactan en prosa por un modelo de escritor local. La respuesta a cada pregunta sobre una escena proviene del plan del código, nunca de un modelo.

Es el complemento de [reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab), que hace
lo mismo para los rompecabezas y los elementos de verificación.

## ¿Por qué escenas con claves creadas con código?

En una escena planificada, la respuesta correcta se deduce mecánicamente de los eventos: quién se movió a dónde, quién podía ver, quién dijo
qué y quién lo escuchó. Alguien que no estaba a la vista cuando se movió un objeto todavía cree que está donde lo vio por última vez. Debido a que esas reglas están en el código, la clave se calcula, no se juzga.

Una *clave creada con código* significa:
- la respuesta proviene del plan del código de la escena;
- está sellada con una semilla privada;
- nunca es escrita o calificada por un modelo.

Cada versión en prosa se verifica mediante código con el plan antes de que se tenga en cuenta.

## Estado: qué se ha probado

Los resultados son conteos en bancos de desarrollo de 160 versiones. Los resultados negativos se presentan primero, porque dieron forma al
diseño.

| Etapa | Qué se probó | Resultado | ¿Adoptado? |
|---|---|---|---|
| 3 | Una lista de las formas de oración aceptadas en el mensaje del escritor | 111/160 versiones pasaron, en comparación con 127/160 para el mensaje de referencia. El escritor no utilizó las formas enumeradas. | **No**, según la regla preestablecida |
| 4 | Formas de oración asignadas por código | 89/160, en comparación con 120/160 para el mensaje de referencia. La puerta de desarrollo falló, por lo que no hubo una ejecución de confirmación. 16 versiones por rama (10%) nunca se generaron, porque el mensaje más el presupuesto de salida excedieron el límite de contexto del escritor. | **No** |
| 5 | Movimientos de código y discurso reescrito (revisión 2) | La ejecución de desarrollo pasó todas las 7 puertas preestablecidas. Primeros borradores: 126/160, en comparación con un mínimo de 112; 132/160 después de la canalización. Los intercambios de código mantuvieron 2,501/2,501. Las reescrituras del discurso mantuvieron 350/358. 0 versiones excedieron el límite de contexto. | Pasó sus puertas de desarrollo |

**Aún no se ha demostrado:**
- no hay una ejecución de confirmación sellada de la etapa 5;
- no hay un modelo entrenado con datos de udlab;
- aún no hay nada sobre si algún modelo comprende a las personas en una escena.

## Qué hay aquí

- **`udlab-gen`**, el generador. Planifica escenarios, ejecuta la canalización de la etapa 5, sella las claves de cada conjunto con una semilla privada
y verifica un conjunto escrito volviéndolo a renderizar y comparándolo.
- **Sondas de acceso directo**, ejecutadas fuera del bucle. Prueban si las características superficiales de una escena revelan la respuesta.
- **El mensaje del escritor** (`prompts/`). El escritor es `qwen3.6:35b`, que se ejecuta localmente a través de Ollama.
- **Un lanzador de Colab** (`scripts/colab/`) para ejecutar el escritor en una GPU alquilada.
- **759 pruebas.**

## Pruébalo sin una GPU

El escritor predeterminado es un simulacro de CPU, por lo que toda la canalización se ejecuta en cualquier máquina:

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15 o posterior; `uv` lo instala si es necesario.
- El primer comando escribe una parte pública (`part_public/`) y un almacén sellado (`sealed_store/`) en `artifacts/demo/`,
una carpeta que Git ignora.
- `uv run udlab-gen --help` enumera todas las opciones.

## Qué permanece privado

Estos permanecen privados, porque publicarlos revelaría las respuestas:
- el banco de pruebas sellado y los bancos de semillas de desarrollo;
- cada almacén sellado y registro de generación;
- los registros de ejecución con sus semillas de plan.

Ninguno de ellos está en este repositorio ni en su historial. Las pruebas utilizan solo semillas sintéticas y escenas inventadas.

## Seguridad y confianza

- **Datos accedidos:** archivos en el `--out-dir` que elijas y el mensaje del escritor. Con `--writer ollama`, también
llama a un servidor local de Ollama.
- **Datos no accedidos:** nada fuera de la carpeta de salida. No hay telemetría y no hay llamadas de red, excepto al
servidor local de Ollama y, si lo ejecutas, al lanzador de Colab.
- **Permisos:** ninguno más allá de la lectura y la escritura de la carpeta de salida. En el modo GPU, detiene el servidor de Ollama que
inició si la memoria de video supera `--vram-halt-mib`.
- Informa sobre los problemas de seguridad según lo que describe [SECURITY.md](SECURITY.md).

## Licencia

El código y el texto propio del repositorio están bajo la [licencia MIT](LICENSE). La definición citada debajo del logotipo es
de Wiktionary y permanece bajo [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). No se utiliza en
ningún conjunto de datos ni mensaje.
