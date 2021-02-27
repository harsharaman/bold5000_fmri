import pickle
import numpy as np
import re
import collections

imageNetRegex = re.compile(r'^(n.*?)_.*?\.JPEG$')
cocoRegex = re.compile(r'^COCO_train2014_0+(.*?).jpg$')


imagenet_label_dict = {}
with open('imagenet_super_categories.txt', 'r') as f:
    for line in f.readlines():
        contents = line.strip().split(' ')
        imagenet_label_dict[contents[0]] = contents[1]

coco_cat_lookup = {}
with open('coco_super_categories.txt', 'r') as f:
    for line in f.readlines():
        contents = line.strip().split(' ')
        coco_cat_lookup[contents[0]] = contents[1]

with open('coco_final_annotations.pkl', 'rb') as f:
    cocoLabels = pickle.load(f)

#print(coco_cat_lookup)
coco_label_dict = {}
for image_id, contents in cocoLabels.items():
    category = str(contents[0].get('category_id'))
    #print(category)
    coco_label_dict[str(image_id)] = coco_cat_lookup.get(category)

#print(coco_label_dict)

#Loop through all stim lists and create labels
# index in training set
# label class categories

global_label_lookup = {key: value for key, value in imagenet_label_dict.items()}
global_label_lookup.update(coco_label_dict)

mega_categories_lookup = {'artifact': 'artifact', 'animal': 'animal', 'food': 'food', 'plant': None, 'scene': 'scene', 'communication': None,
'person': 'person', 'vehicle': None, 'furniture': None, 'kitchen': 'food', 'sports': 'person', 'indoor': None, 'electronic': None, 'accessory': None,
'outdoor': 'scene', 'appliance': None}

classes = {'animal': 0, 'artifact': 1, 'scene': 2, 'person': 3, 'food': 4}

# load stimuli list
subject_labels = {'CSI01': [], 'CSI02': [], 'CSI03': [], 'CSI04': []}
subject_mega_labels_count = {'CSI01': None, 'CSI02': None, 'CSI03': None, 'CSI04': None}

subject_persource_image_count = {'CSI01': None, 'CSI02': None, 'CSI03': None, 'CSI04': None}
subject_persource_label_count = {'CSI01': None, 'CSI02': None, 'CSI03': None, 'CSI04': None}




for subject in subject_labels.keys():
    imageNetCount = 0
    cocoCount = 0
    sceneCount = 0
    imageNetCategoriesCount = collections.Counter()
    cocoNetCategoriesCount = collections.Counter()
    globalCategoriesCount = collections.Counter()

    with open('stim_lists/%s_stim_lists.txt' % subject, 'r') as f:
        stimuli_list = f.read().splitlines()
        for imageFileName in stimuli_list:
            match = imageNetRegex.match(imageFileName)
            if match:
                imageLabel = match.group(1)
                category = imagenet_label_dict.get(imageLabel, None)
                if not category:
                    print("could not find category for image: %s" % imageFileName)
                    continue

                imageNetCategoriesCount.update([category])
                megaCat = mega_categories_lookup.get(category)
                if megaCat:
                    globalCategoriesCount.update([megaCat])

                imageNetCount += 1
                classNum = classes.get(megaCat, -1)
                subject_labels[subject].append(classNum)
                continue

            match = cocoRegex.match(imageFileName)
            if match:
                imageLabel = match.group(1)
                category = coco_label_dict.get(imageLabel)
                #if 'outdoor' in category:
                #
                #     print(imageFileName)

                if not category:
                    print("could not find category for image: %s" % imageFileName)
                    continue

                cocoNetCategoriesCount.update([category])
                megaCat = mega_categories_lookup.get(category)
                if megaCat:
                    globalCategoriesCount.update([megaCat])

                cocoCount += 1
                classNum = classes.get(megaCat, -1)
                subject_labels[subject].append(classNum)

                continue

            globalCategoriesCount.update(['scene'])
            sceneCount += 1
            classNum = classes.get('scene', -1)
            subject_labels[subject].append(classNum)

    assert sum(imageNetCategoriesCount.values()) == imageNetCount
    assert sum(cocoNetCategoriesCount.values()) == cocoCount
    subject_persource_image_count[subject] = [imageNetCount, cocoCount, sceneCount]
    subject_persource_label_count[subject] = [imageNetCategoriesCount, cocoNetCategoriesCount, sceneCount]
    subject_mega_labels_count[subject] = globalCategoriesCount

#print(subject_persource_image_count)
#for subject, counterList in subject_persource_label_count.items():
#    for counter in counterList:
#        print(counter)

#for subject, counter in subject_mega_labels_count.items():
#    print(counter)
for subject, label_list in subject_labels.items():
    # print(label_list)
    print(sum(label_list))
    #print(sum([val for val in label_list if val != -1]))
    #print(len([val for val in label_list if val == -1]))

subject_labels_mask = {'CSI01': [], 'CSI02': [], 'CSI03': [], 'CSI04': []}
# Create masks for data labeled as -1
for subject, labels_list in subject_labels.items():
    labels_array = np.asarray(labels_list)
    subject_labels_mask[subject] = labels_array != -1

print(subject_labels_mask)



