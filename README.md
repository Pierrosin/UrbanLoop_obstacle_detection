# Détection d'obstacles sur les trajets des capsules UrbanLoop

## Objectif du projet

Le but de ce projet est de mettre en place un algorithme permettant de détecter des obstacles sur les rails des capsules UrbanLoop.

En effet, le système de transport en commun UrbanLoop est totalement automatisé et les capsules sont autonomes et sans conducteur.

Il est donc nécessaire de mettre en place un système de sécurité permettant de détecter ces situations dangereuses afin de pouvoir intervenir en actionnant des freins d'urgence par exemple.

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Photos%20non%20analys%C3%A9es/UrbanLoop4.jpg)

## Solution proposée

La méthode mise en oeuvre pour ce projet consiste en une solution d'IA embarquée de vision par ordinateur.

L'idée est de positionner une caméra embarquée sur l'avant de chacune des capsules UrbanLoop et de la connecter à l'unité centrale de la capsule.

Cette caméra filme et analyse en continu et temps réel son environnement et renvoie à l'unité centrale les informations liés à la présence ou l'absence d'obstacles sur les rails.

## Mise en oeuvre

L'implémentation de cette solution est décomposée en deux processus combinés afin de détecter les obstacles sur les rails.

Le premier correspond à la détection d'objets par YOLO et le second correspond à la détection des rails par traitement d'images.

### Détection d'objets par YOLO

Dans un premier temps, chaque image du flux vidéo est analysée par algorithme de détection d'objets YOLOv4 qui donne le type d'objet et sa position dans l'image.

Cet algorithme est associé à un modèle de réseau de neurones qui permet de détecter 80 types de labels différents (humain, animaux, objets...) mais aussi avec mon propre modèle de reconnaissance des capsules UrbanLoop entrainé sur une database d'environ 500 images.

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Photos%20analys%C3%A9es/UrbanLoopYolo.png)


### Détection des rails

Ensuite, l'image est analysée afin de détecter et localiser les deux rails par traitement d'images via l'utilisation de filtres d'équilibre des histogrammes, de flou et de convolution ainsi des courbes de Bézier.

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Filtres%20d%C3%A9tection%20rails/HistogramEqualizationFilterRail.png)

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Filtres%20d%C3%A9tection%20rails/BlurFilterRail.png)

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Filtres%20d%C3%A9tection%20rails/ConvolutionFilterRail.png)

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Filtres%20d%C3%A9tection%20rails/RailDetection.png)

### Détection des obstacles

Enfin, pour chaque image, on peut déterminer si chacun des objets détectés se situe entre les rails en fonction de sa position par rapport aux rails.

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Photos%20analys%C3%A9es/UrbanLoopSafe.png)

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Photos%20analys%C3%A9es/UrbanLoopDanger.png)

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Photos%20analys%C3%A9es/UrbanLoopSafe2.png)

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Photos/Photos%20analys%C3%A9es/UrbanLoopDanger2.png)

Une vidéo de démonstration est disponible : https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/Vid%C3%A9os/D%C3%A9monstrationD%C3%A9tectionObstacleUrbanLoopAnalysed.avi
