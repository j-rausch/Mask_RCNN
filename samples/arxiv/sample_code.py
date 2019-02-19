import os,json

if __name__ == "__main__":
    dataset_mapping_json = 'tables_'+subset+'.json'
    image_root_path = os.path.join(dataset_dir, 'images')
    jsons_root_path = os.path.join(dataset_dir, 'jsons')


    with open(dataset_mapping_json, 'r') as fp:
        docs_and_pages = json.load(fp)

    self.all_ann_json_paths = [] 
    self.all_img_json_paths = []
    self.all_image_paths = []
    self.page_nrs = []
    for doc_id, pages in docs_and_pages:
        for page in pages:
            self.all_ann_json_paths.append(os.path.join(jsons_root_path, doc_id, 'table_anns.json'))
            self.all_img_json_paths.append(os.path.join(jsons_root_path, doc_id, 'imgs.json'))
            img_name = doc_id + "_page{0:0=3d}_feature.png".format(page)
            self.all_image_paths.append(os.path.join(image_root_path, doc_id, img_name))
            self.page_nrs.append(page)
    image_ids = range(len(self.all_image_paths))
