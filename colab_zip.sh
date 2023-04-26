#!/bin/bash
# zip colab_version -r models saves utils audio_classification.ipynb
zip colab_version -r * -x "data*" -x "*.pt" -x "colab_zip.sh"