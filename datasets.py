from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import json
import pandas as pd
import numpy as np
from typing import List,\
                   Dict,\
                   Callable

class VariantDataset(Dataset):
    INFO_QUERY_TEMPLATE =\
        """
        SELECT
          *
        FROM
          porsche_data.variants
        WHERE
          variant_id IN {:s}
        """

    def __init__(self,
                 variant_uids: List,
                 transforms: Callable,
                 path: str,
                 download: bool) -> None:
        """
        dataset for ss variants

        Args:
            variant_uids (List):
                the list of variant_ids for product variants to grab from
                porsche_data.variants

            transforms (Callable):
                function that will take in the query results and process into
                whatever form (eg a vector representation) will be served by
                the dataset when __getitem__ called

            # NOTE: in future, would make to be able to not have to save
            #       locally, and stream directly from DB to the dataset.
            #       doing must save local now to get started somewhere
            path (str):
                where pulled data is going to be stored locally

            download (bool):
                should we download and save the query results, or have we
                already downloaded?
        """
        super().__init__()
        self.variant_uids = variant_uids
        self.path = path
        self.return_image = return_image

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if download:
            print('downloading dataset:')
            from dataradeh20.access.big_query import BigQueryClient
            bq = BigQueryClient()

            print('querying bigquery for data...')
            info_query = self.INFO_QUERY_TEMPLATE\
                    .format(''.join(['(', str(self.variant_uids)[1:-1], ')']))
            variant_info = bq.run_query(info_query)
            print('saving to {:s}...'.format(self.path))
            variant_info.to_json(self.path)
        else:
            assert os.path.exists(self.path),\
                "dataset {:s} does not exist.  change path or set download=True"
        print('loading dataset from {:s}...'.format(self.path))
        variant_info = pd.read_json(self.path)
        self.variant_info = variant_info.set_index('variant_id')

        self.data = self.transforms(self.variant_info)


    def __getitem__(self, index: int) -> Dict:
        uid = self.variant_uids[index]
        return self.variant_info.loc[uid]

    def __len__(self):
        return len(self.variant_uids)


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

    # dataset = VariantDataset(variant_uids=IDS,
    #                          transforms=lambda x: x,
    #                          path='test_data/test_path.json',
    #                          download=True)

    dataset = VariantDataset(variant_uids=IDS,
                             transforms=lambda x: x,
                             path='test_data/test_path.json',
                             download=False)
    data_loader = DataLoader(dataset)



    # TRYING TO UNDERSTAND INFO OF THE DATA

    # maybe constant by affiliate?  trying with linkshare
    df = dataset.variant_info
    linkshare_uids = df[df.affiliate_id == "linkshare"].affiliate_id
    linkshare_inds = np.where([uid in linkshare_uids for uid in IDS])[0]
    lsp = [dataset.__getitem__(ii) for ii in linkshare_inds]

    def parse_fn(lsp):
        json.loads(lsp.raw)['attributeClass']['Product_Type']

    # will fail on the 23rd one - full of errors
    for ii, lsp0 in enumerate(lsp):
        try:
            json.loads(lsp0['raw'])['attributeClass']['Product_Type']
        except:
            print(ii)
            break
