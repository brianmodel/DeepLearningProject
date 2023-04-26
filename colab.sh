#!/bin/bash
[ -d "colab_version" ] && rm -rf colab_version
zip colab_version -r * -x "data*" -x "*.pt" -x "colab_zip.sh"
unzip colab_version.zip -d colab_version
rm colab_version.zip