# Galton's Hoard

Banco de pruebas local para los modelos de lenguaje y de visión de tu ordenador. Plantea a cada modelo tareas cuya respuesta se puede comprobar de forma automática (razonamiento, matemáticas, código Python con pruebas ocultas, extracción de JSON, llamadas a herramientas, seguimiento de instrucciones, búsqueda en textos largos, respuestas con citas, imágenes, traducción, resúmenes y redacción en castellano, además de los casos que añadas tú), guarda cada respuesta con sus tiempos y su uso de memoria, ordena y compara los modelos con intervalos de confianza y pruebas pareadas, avisa cuando un modelo es nuevo o ha cambiado y publica una tabla de rutas que indica qué modelo medido es el mejor para cada tipo de tarea.

Forma parte de la familia Hoard de aplicaciones locales: funciona en tu PC, guarda sus datos en `data/`, se usa por sí sola en el navegador y un asistente puede manejarla por MCP.

[English version](README.md)

## Qué hace

- **Encuentra tus modelos.** Los que ya sirve Ollama, los que sirve llama-server (puertos 8080 a 8090) o aparecen en el registro de Faustus, los archivos GGUF de las carpetas que configures (por defecto `D:\LocalAI\models` si existe) y los modelos de Ollama cuyo archivo está en el disco. De un modelo de Ollama se ofrecen las dos formas de ejecutarlo: por la API de Ollama o en el llama-server propio de Galton con el mismo archivo (la opción por defecto cuando se encuentra el archivo). En Windows la carpeta de Ollama se encuentra con `OLLAMA_MODELS` del entorno o, si la aplicación no se arrancó desde una consola que lo tenga, de las variables de usuario y de equipo del registro; gana la carpeta cuyos `manifests` contienen modelos de verdad. Los alias son solo nombres que un servidor o una aplicación pueden informar (etiquetas de Ollama, el alias de llama-server, el id de `/v1/models`, el nombre de un archivo GGUF); las rutas de archivo y los nombres de blob `sha256-…` quedan en sus propios campos y se quitan al arrancar de los alias guardados por versiones anteriores. Un modelo cuyo archivo cambia se detecta por su huella y sus resultados antiguos se marcan como caducados, nunca se mezclan con los nuevos. Un llama-server que carga un blob de Ollama ejecuta los mismos pesos que la etiqueta de Ollama: ambas entradas responden a los nombres de la otra y la ficha del modelo las lista como mismos pesos. Un servidor que responde a `/v1/models` pero no sabe conversar (sin plantilla de chat, o con un error ante una petición de prueba) se marca una vez como no apto para chat, se desactiva y nunca se anuncia como modelo nuevo. La identidad de un concursante en un servidor compartido es su dirección más lo que el servidor dice que sirve (ids de `/v1/models`, alias y archivo de modelo de `/props`, o el digest de Ollama). Una ejecución lo comprueba antes de que el concursante empiece y otra vez antes de cada caso (la consulta se guarda unos 20 s) y compara el campo `model` de cada respuesta; si alguien reinició el servidor con otro modelo, el concursante se detiene con un error y no se guarda nada medido después del cambio. `models_refresh` marca como "no se sirve ahora" al concursante cuya dirección sirve ya otro modelo: conserva su historial, los formularios de ejecución lo excluyen indicando el motivo y vuelve cuando el servidor lo sirve de nuevo; el modelo nuevo recibe su propio concursante.
- **Mide con tareas comprobables.** 225 casos en 13 suites incluidas, en castellano de España salvo que la tarea trate de otro idioma. Quince tipos de comprobación: texto exacto, contiene, expresión regular, opción múltiple, número (formatos español e inglés, con tolerancia; tras una marca de respuesta como «Respuesta:» lee el resultado de un cálculo de esa línea, así `Respuesta: 2^10 = 1024` vale 1024; si esa línea no tiene cifras lee los números escritos con palabras y los ordinales en castellano e inglés, así `Respuesta: El octavo día` vale 8), equivalencia matemática (el LaTeX como `\(3x^2\ln(x)+x^2\)` se normaliza antes de la comparación simbólica y solo cuenta lo que hay tras el último `=` de la línea de respuesta), JSON con esquema, llamada a herramienta, Python con pruebas ocultas, restricciones de instrucciones, aguja en un texto largo, citas, un modelo juez con rúbrica, una llamada a otra aplicación Hoard y ninguna (el caso se guarda para la arena). Una comprobación de palabras o de restricciones puede limitarse a la respuesta final (`scope: "answer"`: solo el texto tras la última línea «Respuesta:» o «Answer:»), y un campo de un caso de extracción se puede comparar como `contains` o `norm` (ignora una palabra genérica inicial como sala, calle o avda.), de modo que un modelo que dice `sala Magallanes` no suspende por `Magallanes`. Cada caso determinista incluido lleva su respuesta de referencia y una prueba comprueba que la supera con su propia comprobación.
- **Da espacio a los modelos que razonan.** Si un modelo razona se lee de la plantilla de chat de llama-server (`enable_thinking` o `<think>`), de las capacidades que informa Ollama (`thinking`), o se aprende cuando una respuesta llega con contenido de razonamiento. Para ese modelo, salvo que el esfuerzo de la ejecución sea `off`, el límite de salida es el presupuesto de respuesta del caso más `runner.reasoning_tokens` (8192 por defecto, 0 lo desactiva; un esfuerzo mayor pide más) y el tiempo límite crece con él. El esfuerzo `off` envía `enable_thinking: false` a llama-server y `think: false` a Ollama. Un resultado cuya respuesta visible está vacía porque se alcanzó el límite se guarda como `truncated` (cortado); un modelo del que no se sabía que razona y que lo demuestra se vuelve a preguntar una vez con ese espacio y se recuerda. La página de la ejecución y la clasificación muestran cuántos resultados se cortaron por modelo, y la clasificación, la tabla de rutas y `recommend` avisan cuando más del 5 % de los resultados de un modelo en una categoría están cortados (el presupuesto era pequeño y la nota infravalora al modelo).
- **Ejecuta los modelos en las GPU que permitas.** Un GGUF se ejecuta en un llama-server que arranca Galton en un puerto libre entre 8091 y 8099, con una reserva de la memoria estimada (tamaño del archivo más caché KV más margen), en una GPU permitida o repartido entre varias, y al terminar se mata todo el árbol de procesos. Las GPU 0 y 1 están reservadas por defecto y las permitidas son la 2 y la 3; no se pueden permitir las reservadas sin una confirmación explícita. Los servidores que ya están en marcha se usan tal cual, pero nunca mientras los usa otra persona: si uno está ocupado, ese modelo pasa al estado `waiting_server` (la página de la ejecución y `run_status` muestran cuánto lleva esperando) y la medición empieza cuando el servidor lleva en calma `runner.idle_grace_s` (20 s por defecto); solo se rinde pasado `runner.wait_idle_max_s` (una hora por defecto, 0 espera sin límite; `wait_s` de la ejecución lo sustituye). Entre dos preguntas vuelve a mirar y hace la misma pausa si alguien ha empezado una conversación (llama-server muestra sus slots; en Ollama la actividad se deduce de `expires_at` en `/api/ps`, así que una petición aún en curso no se ve); un modelo de Ollama que no esté cargado solo se carga si lo activas.
- **Recurre a la CPU con los archivos pequeños.** Cuando un GGUF no cabe ahora en ninguna GPU permitida (todas ocupadas por otra carga, o ninguna de las permitidas existe) y su archivo pesa como máximo `runner.cpu_max_gb` (4 GB por defecto), Galton arranca su propio llama-server con `-ngl 0`, `CUDA_VISIBLE_DEVICES=""` (no toca nunca una GPU), `-t` = núcleos físicos menos 2 (mínimo 2) y sin reserva, en lugar de esperar una GPU: un modelo que puede usar una GPU libre sigue usándola. El ajuste de ejecución `device` es `auto` (este comportamiento), `gpu` (nunca la CPU: exactamente el comportamiento anterior, con su espera y su error) o `cpu` (siempre; el límite de tamaño no se aplica). `runner.cpu_fallback` (activado por defecto) desactiva la elección automática. La CPU también necesita memoria: con poca RAM libre se conserva el error de GPU de siempre. El plan dice dónde correría cada modelo («CPU (no hay GPU libre)»), la tarjeta de la ejecución y los resultados se marcan como CPU, y una velocidad de CPU nunca se mezcla con las de GPU: para un modelo, la clasificación, `compare` y las rutas usan sus mediciones de GPU si las tiene (la cifra de CPU se muestra aparte) y, si no, las de CPU con una etiqueta «CPU»; una velocidad solo de CPU no cuenta en el peso de la velocidad del ranking. La vigilancia nunca pone una ejecución en segundo plano en la CPU.
- **Dice cuánta confianza hay.** Tasa de acierto con intervalo de Wilson al 95 %; puntuación media con intervalo bootstrap al 95 % con semilla, ponderada por el peso de cada caso; dos modelos se comparan en los casos que comparten con la prueba exacta de McNemar y un bootstrap pareado, y el veredicto es mejor, peor, sin diferencia clara o sin datos. La velocidad (tokens por segundo, tiempo hasta el primer token, tiempo de carga) y la memoria salen de las ejecuciones, no de estimaciones.
- **Publica la tabla de rutas.** Para cada categoría de tarea los candidatos son los modelos activos con resultados recientes de su archivo actual; los puntos de acceso remotos quedan fuera salvo que se permitan; los modelos demasiado lentos o demasiado grandes se excluyen; el resto se ordena por el límite inferior del intervalo al 95 %, con la velocidad de generación como desempate. Cada elección lleva una frase que la explica. La tabla se escribe de forma atómica en `routes.json` (`HOARD_ROUTES_FILE` o `~/.hoard/routes.json`) y Hoard Link la lee, de modo que las demás aplicaciones de la familia usan el mejor modelo medido. Una categoría con menos casos comprobados de los que pide la política no se publica nunca. Antes de publicar se ve qué cambiaría.
- **Vigila.** Los modelos nuevos o cambiados se miden con la suite rápida cuando no molesta a nadie (un servidor cargado y libre desde hace 10 minutos, o un GGUF que cabe en una GPU permitida libre; nunca en horas de silencio, de 01:00 a 08:00 por defecto). Tras cada ejecución compara el modelo con los resultados que tenía antes de que cambiara su archivo y, si ha empeorado de forma significativa, crea un aviso y emite el evento de la familia `galton.regression`.
- **Arena.** Dos respuestas anónimas al mismo enunciado, tomadas de ejecuciones guardadas (no gasta tiempo de modelo), tú votas y se calculan puntuaciones Bradley-Terry por categoría.
- **Tus propias suites.** Crea una suite, añade casos a mano o importa JSONL o CSV (las columnas pueden estar en castellano o en inglés; basta una comprobación en una línea como `contains:uno|dos` o `number:42`). Las suites incluidas son de solo lectura y se pueden duplicar. `case_try` ejecuta un caso en un modelo sin guardar nada.
- **Juez.** Un modelo que eliges puntúa las respuestas abiertas de 0 a 10 según una rúbrica. Nunca se puntúa a sí mismo en silencio (esos resultados se marcan `self_judged`), las notas se guardan según lo que vio el juez, un juez que es el propio modelo medido puntúa solo después de su último caso (así puntuar nunca retrasa una pregunta cronometrada) y las respuestas quedan pendientes, no suspendidas, mientras el juez no está disponible.

## Pantallas

En castellano por defecto, en inglés con un clic, oscuro.

- **Panel**: tabla de rutas con nota, intervalo, velocidad y memoria; la ejecución en curso; avisos; las GPU con sus reservas; botones para medir lo nuevo y publicar las rutas.
- **Modelos**: cada modelo con tipo, familia, tamaño, cuantización, contexto, si cabe en 16 GB, última medición y marcas de caducado; activar, renombrar, añadir alias, añadir un archivo o un servidor.
- **Pruebas**: suites y casos con sus comprobaciones; crear, duplicar, importar y probar un caso.
- **Ejecutar**: elegir suites y modelos, ver el plan (casos, memoria, dónde correría cada modelo, problemas), empezar, seguir el progreso caso a caso, cancelar (el mensaje dice qué se detiene: solo el llama-server que Galton arrancó él mismo; un servidor compartido no se toca nunca), descartar una ejecución terminada indicando el motivo (queda en el historial con una etiqueta, pero ninguna estadística, clasificación, ruta, comparación ni comprobación de regresiones la usa; restaurar lo deshace) y leer cada respuesta con el detalle de la comprobación.
- **Clasificación**: ranking por categoría o suite con barras de intervalo, columnas de velocidad y memoria, y la comparación pareada de dos modelos con la lista de casos en que difieren.
- **Arena**: votos a ciegas y puntuaciones.
- **Rutas**: la tabla de rutas, la diferencia con lo publicado y el historial de publicaciones.
- **Ajustes**: GPU permitidas, ruta de llama-server, juez, límites de ejecución, vigilancia de regresiones, política de rutas, idioma.

## Qué no hace

- Mide lo que comprueban sus casos. Una nota alta en una suite dice poco de tareas que no están en ella; añade casos propios para el trabajo que te importa.
- El llama-server propio de Galton solo usa la CPU con archivos de hasta `runner.cpu_max_gb` (o cuando una ejecución indica `device: cpu`), y su velocidad no es comparable con la de las GPU. Un GGUF mayor que no cabe ni sumando las GPU permitidas se declara imposible con las cifras, no se intenta.
- Necesita `llama-server` (llama.cpp) instalado para ejecutar GGUF por su cuenta; sin él solo se miden servidores que ya estén en marcha.
- Las notas del juez valen lo que valga el modelo juez que elijas, y una respuesta que espera al juez no cuenta hasta que se puntúa. Una petición que falla porque el servidor se rompe se guarda con su error y queda fuera de la nota; nunca cuenta como respuesta equivocada.
- Los casos de Python ejecutan código escrito por el modelo en un intérprete aparte, con tiempo límite y entorno limpio. No es un entorno aislado de seguridad; desactívalo con el ajuste `checks.allow_code_execution` si no lo quieres.
- Los casos de visión necesitan un modelo con su archivo de proyector; los modelos sin él se saltan y se guarda el motivo.
- No entrena, cuantiza ni descarga modelos, y no llama a puntos de acceso remotos salvo que los actives modelo a modelo.
- El espacio que se da al razonamiento es un límite, no una medida de cuánto razona un modelo; uno que necesita más de `runner.reasoning_tokens` sigue cortándose y se muestra como cortado.
- Con pocas muestras los intervalos son anchos. La aplicación los muestra en lugar de ocultarlos y nunca publica una categoría con pocos casos.

## Instalación

Requisitos: Python 3.11 o superior (probado en 3.11 y 3.13), Node 22 solo para reconstruir la interfaz, un binario de llama-server para los GGUF y Ollama si lo usas.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # Windows; venv/bin/python en otros sistemas
```

La interfaz viene compilada en `galton_hoard/static`. Para reconstruirla: `npm install && npx vite build`.

## Ejecución

```bash
venv/Scripts/python -m galton_hoard                           # http://127.0.0.1:5201
python scripts/launch.py                                      # puerto libre, abre el navegador
```

El modo demostración enseña toda la aplicación sin tocar hardware: `GALTON_FAKE=1 python -m galton_hoard` usa GPU inventadas y tres modelos que responden desde una tabla, y publica en `data/routes-demo.json`.

La interfaz está en castellano o en inglés; los mensajes del servidor (errores con su pista, líneas del plan, avisos de las ejecuciones, notificaciones) le llegan como una clave estable más parámetros y se formatean en el idioma elegido, mientras que los asistentes reciben siempre la frase en inglés. Sin consola (`pythonw`) los mensajes de arranque van solo al registro.

Entorno: `GALTON_PORT` (5201), `GALTON_DATA_DIR`, `PORT_STRICT=1`, `GALTON_ALLOWED_HOSTS`, `GALTON_HTTP_TIMEOUT_S`, `GALTON_SCHEDULER=0` (sin tareas de fondo), `GALTON_OFFLINE=1` (sin búsqueda de modelos), `GALTON_FAKE=1` (demostración), `HOARD_ROUTES_FILE`, y `GALTON_FAUSTUS_TOKEN` en `.env` (véase `.env.example`). Todo lo demás es un ajuste guardado en `data/galton.db` que se cambia en la aplicación.

## Asistentes (MCP)

`mcp_server.py` es un puente MCP por stdio llamado `galton-hoard`. No abre la base de datos: reenvía cada llamada a la aplicación en marcha con el token de `data/mcp-token` y arranca la aplicación si no responde. `faustus-plugin.json` describe la aplicación, su comprobación de salud y el puente para Faustus y el Hoard Hub.

Herramientas (43): `galton_overview`, `galton_status`, `models_list`, `models_refresh`, `model_get`, `model_add`, `model_update`, `model_remove`, `suites_list`, `suite_get`, `case_get`, `suite_create`, `suite_update`, `suite_duplicate`, `suite_remove`, `case_add`, `case_update`, `case_remove`, `cases_import`, `case_try`, `run_plan`, `run_start`, `run_status`, `run_cancel`, `run_discard`, `run_restore`, `runs_list`, `run_results`, `measure_new`, `judge_run`, `leaderboard`, `compare`, `recommend`, `routes_get`, `routes_publish`, `arena_next`, `arena_vote`, `arena_ratings`, `settings_get`, `settings_set`, `gpu_status`, `notices_list`, `housekeeping_run`. Argumentos en [docs/API.md](docs/API.md).

Eventos en el bus de la familia: `galton.routes.updated` (la tabla publicada ha cambiado), `galton.regression` (un modelo ha empeorado) y `galton.models.updated` (modelos nuevos, cambiados o desaparecidos).

## Datos y privacidad

Todo vive en `data/` (o `GALTON_DATA_DIR`): `galton.db` (modelos, suites, ejecuciones, resultados, ajustes), `images/` (imágenes de tus casos), `logs/` (un registro de llama-server por modelo y `galton.log`, el registro rotativo de la propia aplicación, que es lo que queda cuando se arranca sin consola), `cache/`, `servers.json` (los procesos llama-server que arrancó Galton, para que un fallo no deje un modelo cargado: se detienen en el siguiente arranque), `mcp-token` y `url`. Nada sale del ordenador: la aplicación solo escucha en 127.0.0.1, rechaza peticiones de otros orígenes y solo llama a servidores locales salvo que actives un punto de acceso remoto para un modelo. Los enunciados y respuestas de tus propios casos se guardan tal cual; no pongas secretos en ellos. Copia la carpeta para hacer una copia de seguridad de la aplicación.

## Desarrollo

```bash
python -m pytest                           # sin red, sin GPU, sin llama-server
python scripts/gen_api_doc.py              # regenera docs/API.md tras cambiar una herramienta (un test falla si está desactualizado)
npx vite build                             # reconstruye la interfaz en galton_hoard/static
```

Véanse [AGENTS.md](AGENTS.md) y [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Los tests usan un mundo simulado (GPU y reservas simuladas, modelos que responden con las respuestas de referencia, un lanzador simulado) y HTTP simulado; ninguno toca hardware real.

## Licencia

MIT
