from pycocotools.coco import COCO


dataDir='G:\Data\coco2017\/annotations_trainval2017'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
anns = coco.loadAnns(coco.getAnnIds())
imgs = coco.loadImgs(coco.getImgIds())

labels_ = {str(img['id']):[] for img in imgs}
for ann in anns:
    id = str(ann['image_id'])
    if id in labels_:
        labels_[id].append(ann['bbox'])
imgs = {str(img['id']):img['file_name'] for img in imgs}
labels = {}
for key, value in labels_.items():
    labels[imgs[key]] = value
del labels_



print()