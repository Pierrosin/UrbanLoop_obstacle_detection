# Détection d'obstacles sur les trajets des capsules UrbanLoop

## Objectif du projet

Le but de ce projet est de mettre en place un algorithme permettant de détecter des obstacles sur les rails des capsules UrbanLoop.

En effet, le système de transport en commun UrbanLoop est totalement automatisé et les capsules sont autonomes et sans conducteur.

Il est donc nécessaire de mettre en place un système de sécurité permettant de détecter ces situations dangereuses afin de pouvoir intervenir en actionnant des freins d'urgence par exemple.

![Cover](https://github.com/Pierrosin/UrbanLoop_obstacle_detection/blob/master/UrbanLoop4.jpg)

## Solution proposée

La méthode mise en oeuvre pour ce projet consiste en une solution d'IA embarquée de vision par ordinateur.

L'idée est de positionner une caméra embarquée sur l'avant de chacune des capsules UrbanLoop et de la connecter à l'unité centrale de la capsule.

Cette caméra filme et analyse en continu et temps réel son environnement et renvoie à l'unité centrale les informations liés à la présence ou l'absence d'obstacles sur les rails.

## Mise en oeuvre

L'implémentation de cette solution est décomposée en deux processus combinés afin de détecter les obstacles sur les rails.

Le premier correspond à la détection d'objets par YOLO et le second correspond à la détection des rails par traitement d'images.

### Détection d'objets par YOLO

