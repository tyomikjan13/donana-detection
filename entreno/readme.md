Decisiones sobre los datos:

División por ráfagas: las cámaras disparan varias fotos casi iguales seguidas. Si se reparten al azar, la misma escena acaba en train y en test y el resultado sale inflado. Por eso agrupo las fotos de la misma cámara separadas menos de 60 s y reparto ráfagas enteras, manteniendo la proporción de cada especie. En la prueba ninguna ráfaga quedó repartida entre train y test.
Una caja por foto: uso la caja de MegaDetector con más confianza y descarto las fotos donde aparece una segunda caja fiable, porque probablemente hay otro animal sin etiquetar. Con tu CSV quedan unas 5.190 fotos de 5.411.
Clases raras agrupadas: lince (5 fotos), gineta, meloncillo y caballo pasan a Otros. Se controla con MIN_IMAGES_PER_CLASS.

Tiempo: con Faster R-CNN ResNet50 a 1024 px, cada época con el dataset triplicado tarda del orden de 25–30 min en una T4. Por eso el notebook de aumentación usa por defecto el modelo MobileNet (faster_rcnn_v3), unas cuatro veces más rápido. Cuando sepas qué aumentación funciona mejor, puedes repetirla con el modelo grande en 02_train.

Las cajas de referencia son las de MegaDetector, no anotaciones a mano. Por eso el mAP mide lo bien que el modelo reproduce esas cajas; para comparar aumentaciones entre sí sigue sirviendo.
