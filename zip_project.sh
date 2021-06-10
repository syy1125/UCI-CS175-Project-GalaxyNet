rm -f project.zip

mkdir -p project/src
mkdir -p project/samples
cp dataloader.py knn.py output_layer.py pytorch_init.py pytorch_preprocess.py training.py transforms.py variable_stride.py project/src/
cp project.ipynb project.html effnet-b2-full-best.pt sample_solutions.csv project/
cp project_readme.md project/README.md
cp samples/* project/samples/


zip -r project.zip project/*
rm -rf project