# Prueba de gráficos en tu computador

Gracias por ayudar. Este kit prueba cómo se ve y qué tan rápido va un motor de juegos en tu tarjeta gráfica. No
instala nada y no cambia nada de tu sistema. No envía nada por su cuenta: al final deja **un archivo zip** y
eres tú quien nos lo envía.

## Antes de empezar

1. **Actualiza el controlador de tu tarjeta gráfica** a la versión más reciente:
   - NVIDIA: versión 580 o más nueva (GeForce Experience, la app de NVIDIA, o nvidia.com).
   - AMD en Windows: la versión más reciente de AMD Software (Adrenalin).
   - AMD en Linux: el Mesa más reciente que tenga tu distribución.
2. **Solo en Windows:** instala el «Microsoft Visual C++ Redistributable» para x64 (2015–2022) desde la página
   oficial de Microsoft: <https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist>. Si ya lo tienes,
   el instalador te lo dice.
3. Cierra los juegos y programas pesados. Deja el computador **enchufado** si es un portátil.
4. Si puedes, usa una pantalla de 2560×1440 o más. Con una más pequeña algunas pruebas no se pueden hacer; no pasa
   nada, quedan anotadas.

## Cómo se ejecuta

**Windows**

1. Descomprime `gfx-prueba-windows.zip` en una carpeta (por ejemplo, en el Escritorio). Clic derecho →
   «Extraer todo».
2. Abre la carpeta y haz doble clic en **`PROBAR.bat`**.
3. Si Windows dice «Windows protegió su PC», pulsa **«Más información» → «Ejecutar de todas formas»**. Aparece
   porque el programa no está firmado, no porque haya un problema.

**Linux**

1. Descomprime: `tar xzf gfx-prueba-linux.tar.gz`
2. Ejecuta: `gfx-prueba-linux/probar.sh`

Hace falta un Linux con glibc 2.35 o más nueva (Ubuntu 22.04 o posterior, o equivalente).

## Qué vas a ver

- Una ventana de texto que dice qué está haciendo y cuánto falta. Al principio da una **estimación del tiempo**.
- Muchas ventanas que se abren, muestran una habitación iluminada, y se cierran solas. **No las toques ni cierres:**
  cada una es una medición. Puedes mirar, que es parte de la prueba.
- Cada tarjeta gráfica del computador se prueba por separado (también la integrada del procesador, si la hay, más
  rápido y en pequeño).
- **Cuánto tarda:** depende de la tarjeta; espera entre 30 minutos y 1 hora y media. La estimación del principio es
  la mejor guía.
- Al final de cada tarjeta, la imagen cambia de calidad muchas veces seguidas. Fíjate si se cierra o se queda en
  negro.

## Las cinco preguntas

Al terminar, el programa te pide abrir la carpeta **`para-mirar`** (dentro de la carpeta de resultados) y responde
cinco preguntas con **s** (sí), **n** (no) o **?** (no sé):

1. ¿Estelas o «fantasmas» detrás de los objetos mientras la cámara gira?
2. ¿Luz que atraviesa las paredes?
3. ¿Sombras que parpadean con la cámara quieta?
4. ¿Puntitos sueltos, negros o muy brillantes?
5. ¿Se cerró o se quedó en negro al cambiar de calidad muchas veces?

Si no estás seguro, responde **?**. Es mejor que adivinar.

## Qué nos envías

Un solo archivo que queda junto al programa: **`resultados-<fecha>.zip`** (unos 50–100 MB). Mándalo por el medio
que te resulte cómodo (WeTransfer, Google Drive, etc.).

**Si algo falla o se cierra a mitad de camino**, envía igual el zip que haya: se guarda después de cada paso, y el
primero se escribe al minuto de empezar justamente para esto.

## Qué se recoge y qué no

**Sí se recoge:** el nombre de tu tarjeta gráfica y la versión de su controlador, el sistema operativo y su versión,
los tiempos de cada imagen, los registros (logs) del programa, imágenes de la escena de prueba y tus cinco
respuestas.

**No se recoge:** tu nombre, tu nombre de usuario, el nombre del computador, tus carpetas, tus archivos, números de
serie ni tu dirección IP. Antes de cada zip el programa borra de los textos tu carpeta personal, tu usuario y el
nombre del equipo. Puedes abrir el zip y mirar todo antes de enviarlo: son textos e imágenes.

## Si tienes dudas

- **¿Puedo usar el computador mientras tanto?** Mejor no: cualquier otra cosa cambia los tiempos.
- **¿Puedo pararlo?** Sí: cierra la ventana de texto. Envía el zip que haya quedado.
- **¿Lo tengo que borrar después?** Cuando quieras: borra la carpeta y listo. Fuera de ella solo quedan las
  cachés normales del controlador gráfico y algún registro temporal de NVIDIA, como con cualquier juego.
