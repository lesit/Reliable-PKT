import os
import torch
import torch.utils.data as Data
import numpy as np
import gc

def get_train_data_loader(batch_size, train_data, valid_data, max_seq_len, logger):
    if train_data.dtype != np.float32 or valid_data.dtype != np.float32:
        logger.info(f"loaded data type is not float32: {train_data.dtype}, {valid_data.dtype}")
        legacy_train_data = train_data
        legacy_test_data = valid_data
        train_data = legacy_train_data.astype(np.float32)
        valid_data = legacy_test_data.astype(np.float32)

        del legacy_train_data
        del legacy_test_data
        gc.collect()

    logger.info(f"get_train_data_loader. max_seq_len:{max_seq_len}")

    train_data = train_data[:, -max_seq_len:, :]
    valid_data = valid_data[:, -max_seq_len:, :]
    
    train_tensor = torch.from_numpy(train_data)
    test_tensor = torch.from_numpy(valid_data)

    train_loader = Data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_test_data_loader(batch_size, test_data, max_seq_len, logger):
    if test_data.dtype != np.float32:
        logger.info(f"loaded data type is not float32: {test_data.dtype}")
        legacy_test_data = test_data
        test_data = legacy_test_data.astype(np.float32)

        del legacy_test_data
        gc.collect()

    logger.info(f"get_test_data_loader. max_seq_len:{max_seq_len}")

    test_data = test_data[:, -max_seq_len:, :]

    test_tensor = torch.from_numpy(test_data)
    test_loader = Data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return test_loader


