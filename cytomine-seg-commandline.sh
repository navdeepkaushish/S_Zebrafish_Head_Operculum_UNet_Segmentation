#demo-segmentation-tissue lung1 image
#python cytomine-stardist.py -p 528050 -i 528132 -r 154121648 -t 154122471 -u 701 -wd "/home/maree/Documents/_bds/stardis/cytomine/"
#python cytomine-stardist.py -p 528050 -i 528132 -r 154121648 -t 154122471 -u 263676 -wd "/tmp/cytomine/" -prob_t 0.5  #jsnow
python cytomine-stardist.py -p 528050 -i 528460 -r 154121648 -t 154122471 -u 263676 -wd "/tmp/cytomine/" -prob_t 0.5  #jsnow


#demo-segmentation-tissue cmu image
#python cytomine-stardist.py -p 528050 -i 128010598 -r 154121648 -t 154122471 -u 701 -wd "/home/maree/Documents/_bds/stardis/cytomine/" -prob_t 0.4 -nms_t 0.5

#thyroid
#python cytomine-stardist.py -p 77150529 -i 77150809 -r 154005477 -t 35777309 -u 701 -wd "/home/maree/Documents/_bds/stardis/cytomine/"

#anapath longuespee
#python cytomine-stardist.py -p 30902939 -i 30903021 -r 154367038 -t 31879372 -u 701 -wd "/home/maree/Documents/_bds/stardis/cytomine/" #testroi


#demo-counting
#python cytomine-stardist.py -p 138066737 -i 138066995 -r 138066725 -t 154377928 -u 701 -wd "/home/maree/Documents/_bds/stardis/cytomine/"

python run.py 
