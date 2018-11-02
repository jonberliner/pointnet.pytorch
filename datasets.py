from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
import pandas as pd
import numpy as np
from typing import List,\
                   Dict

class ProductDataset(Dataset):
    ATTRIBUTES_QUERY_TEMPLATE =\
        """
        SELECT
          *
        FROM
          porsche_data.variants
        WHERE
          id IN {:s}
        LIMIT
          10000
        """

    def __init__(self,
                 product_uids: List,
                 path: str,
                 download: bool,
                 return_image: bool=False) -> None:
        """
        dataset for ss products
        """
        # NOTE: the unique id "id" in porsche_data.variants, not "product_id"
        self.product_uids = product_uids
        self.path = path
        self.return_image = return_image

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if download:
            print('downloading dataset:')
            from dataradeh20.access.big_query import BigQueryClient
            bq = BigQueryClient()

            print('querying bigquery for data...')
            attributes_query = self.ATTRIBUTES_QUERY_TEMPLATE\
                    .format(''.join(['(', str(self.product_uids)[1:-1], ')']))
            product_attributes = bq.run_query(attributes_query)
            print('saving to {:s}...'.format(self.path))
            product_attributes.to_json(self.path)
        else:
            assert os.path.exists(self.path),\
                "dataset {:s} does not exist.  change path or set download=True"
        print('loading dataset from {:s}...'.format(self.path))
        product_attributes = pd.read_json(self.path)
        self.product_attributes = product_attributes.set_index('variant_id')


    def __get_item__(self, index: int) -> Dict:
        uid = self.product_uids[index]
        return self.product_attributes.loc[uid]

    def __len__(self):
        return len(self.product_uids)


if __name__ == '__main__':
    from dataradeh20.access.big_query import BigQueryClient

    IDS_QUERY =\
        """
        SELECT
          variant_id
        FROM
          porsche_data.variants
        LIMIT
          10000
        """

    bq = BigQueryClient()
    IDS = bq.run_query(IDS_QUERY).values.ravel().tolist()

    dataset = ProductDataset(product_uids=IDS,
                             path='test_data/test_path.json',
                             download=True)

    dataset = ProductDataset(product_uids=IDS,
                             path='test_data/test_path.json',
                             download=False)



    # TRYING TO UNDERSTAND ATTRIBUTES OF THE DATA

    # maybe constant by affiliate?  trying with linkshare
    df = dataset.product_attributes
    linkshare_uids = df[df.affiliate_id == "linkshare"].affiliate_id
    linkshare_inds = np.where([uid in linkshare_uids for uid in IDS])[0]
    lsp = [dataset.__get_item__(ii) for ii in linkshare_inds]

    def parse_fn(lsp):
        json.loads(lsp.raw)['attributeClass']['Product_Type']

    # will fail on the 23rd one - full of errors
    for ii, lsp0 in enumerate(lsp):
        try:
            json.loads(lsp0['raw'])['attributeClass']['Product_Type']
        except:
            print(ii)
            break
