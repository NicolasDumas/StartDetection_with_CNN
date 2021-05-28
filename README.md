# StartDetection_with_CNN

Contexte :  Intégrer un CNN dans la détection de départ de course pour améliorer la robustesse de l’algorithme précédent.

Le réseau doit classer un son dans l’une des deux catégories suivantes : 
“Contient un bip”
“Ne contient pas de bip”

Les sons en entrée durent 0.5 secondes, durée à réduire pour améliorer la précision.

Deux méthodes implémentées :

Réseau CNN 2D en utilisant le spectrogramme de l’audio (audio_classification.ipynb)
Réseau CNN 1D en utilisant les mêmes 136 features que le premier algorithme (audio_classification_136features.ipynb)


