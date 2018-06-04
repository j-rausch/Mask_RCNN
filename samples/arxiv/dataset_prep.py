import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from sklearn.cross_validation import train_test_split
import numpy
import json

def main():
    image_root_path = '/mnt/ds3lab-scratch/arxiv/25000_split/images'
    jsons_root_path = '/mnt/ds3lab-scratch/arxiv/25000_split/jsons'
    #table_jsons_root_path = '/mnt/ds3lab-scratch/arxiv/25000_split/table_jsons'
    log_file_path = os.path.join(jsons_root_path, 'labels_created_log.txt')

    with open(log_file_path) as f:
        files = [list(x.split('; ')) for x in f.readlines()]
    filtered_file_ids = []
    for x in files:    
        if len(x) < 4:
            continue
        elif x[2]=='success':
            filtered_file_ids.append(x[1].split('id:')[1])
    filtered_file_ids = sorted(set(filtered_file_ids))

    ann_json_paths = [[x, os.path.join(jsons_root_path, x, 'anns.json')] for x in filtered_file_ids]

    docs_with_tables = set()
    doc_pages_with_tables = dict()
    for i, [doc_id, ann_json_path] in enumerate(ann_json_paths):
        with open(ann_json_path, 'r') as fp:
            anns = json.load(fp)
        anns_by_id = dict()
        all_table_anns = []
        table_ids = []
        table_children_ids = []
        for ann in anns:
            anns_by_id[ann['id']] = ann
            if ann['category_name'] == 'table':
                if doc_id not in doc_pages_with_tables:
                    doc_pages_with_tables[doc_id] = set()
                doc_pages_with_tables[doc_id].add(ann['page'])

                table_ids.append(ann['id'])
                table_children_ids += ann['children']

        nested_children_queue = []
        nested_children_queue += table_children_ids
        while len(nested_children_queue) > 0:
            next_child_id = nested_children_queue.pop(0)
            if next_child_id in anns_by_id:
                next_child_ann = anns_by_id[next_child_id]
            if len(next_child_ann['children']) > 0:
                table_children_ids += next_child_ann['children'] 
                nested_children_queue += next_child_ann['children'] 

        all_table_and_children_ids = table_children_ids+table_ids
        all_table_anns = [anns_by_id[x] for x in all_table_and_children_ids if x in anns_by_id]

        table_ann_json_path =  ann_json_path.replace('anns.json','table_anns.json')
        with open(table_ann_json_path, 'w') as fp:
            json.dump(all_table_anns, fp, indent=1)
        
        if i%100 == 0:
            logger.info("currently at doc nr {}".format(i))


    docs_with_tables = list(doc_pages_with_tables.keys())
#
#        table_anns = [x for x in anns if x['category_name'] == 'table']
#        if len(table_anns) > 0:
#            docs_with_tables.append(doc_id)


    ids_train, ids_test = train_test_split(docs_with_tables,test_size=0.15,random_state=42)        
    id_with_pages_train = []
    for id_train in ids_train:
        id_with_pages_train.append([id_train,  sorted(list(doc_pages_with_tables[id_train]))])

    id_with_pages_test = []
    for id_test in ids_test:
        id_with_pages_test.append([id_test, sorted(list(doc_pages_with_tables[id_test]))])


#, ids_test = train_test_split(docs_with_tables,test_size=0.2)        
 
    print('number of valid documents: {}'.format(len(filtered_file_ids)))
    print('number of valid documents with tables: {}'.format(len(docs_with_tables)))
    
    with open('tables_train.json', 'w') as fp:
        json.dump(id_with_pages_train, fp, indent=1)
    
    with open('tables_test.json', 'w') as fp:
        json.dump(id_with_pages_test, fp, indent=1)

if __name__ == "__main__":
    main()
