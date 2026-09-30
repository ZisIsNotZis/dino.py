#!/bin/env python
from sys import argv

from gguf import GGUFReader, GGUFWriter, GGUFValueType, ReaderField, GGMLQuantizationType


def get(f: ReaderField):
    a = [f.parts[i].tobytes().decode() if f.types[-1] == GGUFValueType.STRING else f.parts[i].item() for i in f.data]
    return a if f.types[0] == GGUFValueType.ARRAY else a[0]


for arg in argv[1:]:
    g = GGUFReader(arg)
    f = GGUFWriter(arg.rsplit('-', 1)[0] + '-f16.gguf', get(g.fields['general.architecture']))
    for i in g.fields.values():
        f.add_key_value(i.name, get(i), *i.types)
    for i in g.tensors:
        j = i.data.astype('e') if i.data.dtype == 'f' else i.data
        print(i.name, j.shape, i.data.dtype, i.tensor_type.name, j.dtype)
        f.add_tensor(i.name, i.data.astype('e') if i.tensor_type == GGMLQuantizationType.F32 else i.data)
    f.write_header_to_file()
    f.write_kv_data_to_file()
    f.write_tensors_to_file()
    f.close()
