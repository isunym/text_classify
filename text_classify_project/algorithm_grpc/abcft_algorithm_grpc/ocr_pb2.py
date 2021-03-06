# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ocr.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ocr.proto',
  package='com.abcft.pdfextract.rpc.ocr',
  syntax='proto3',
  serialized_pb=_b('\n\tocr.proto\x12\x1c\x63om.abcft.pdfextract.rpc.ocr\"C\n\tRpcStatus\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x12\n\nerror_code\x18\x02 \x01(\x05\x12\x11\n\terror_msg\x18\x03 \x01(\t\"\x89\x02\n\nOcrRequest\x12\x0e\n\x04\x64\x61ta\x18\x01 \x01(\x0cH\x00\x12\r\n\x03url\x18\x02 \x01(\tH\x00\x12\x44\n\x06params\x18\x03 \x03(\x0b\x32\x34.com.abcft.pdfextract.rpc.ocr.OcrRequest.ParamsEntry\x12\x12\n\nimage_name\x18\x04 \x01(\t\x12\x14\n\x0cneed_denoise\x18\x05 \x01(\x08\x12\x1a\n\x12need_denoise_image\x18\x06 \x01(\x08\x12\x18\n\x10need_parse_title\x18\x07 \x01(\x08\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x42\x07\n\x05image\"K\n\x0f\x42\x61tchOcrRequest\x12\x38\n\x06images\x18\x01 \x03(\x0b\x32(.com.abcft.pdfextract.rpc.ocr.OcrRequest\"S\n\tTextReply\x12\x37\n\x06status\x18\x01 \x01(\x0b\x32\'.com.abcft.pdfextract.rpc.ocr.RpcStatus\x12\r\n\x05texts\x18\x02 \x03(\t\"X\n\x03\x42ox\x12\x10\n\x08\x63\x65nter_x\x18\x01 \x01(\x05\x12\x10\n\x08\x63\x65nter_y\x18\x02 \x01(\x05\x12\r\n\x05width\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\x12\x0e\n\x06rotate\x18\x05 \x01(\x05\"\x94\x02\n\x07\x45lement\x12?\n\x04type\x18\x01 \x01(\x0e\x32\x31.com.abcft.pdfextract.rpc.ocr.Element.ElementType\x12/\n\x04\x62\x62ox\x18\x02 \x01(\x0b\x32!.com.abcft.pdfextract.rpc.ocr.Box\x12\x0c\n\x04text\x18\x03 \x01(\t\"\x88\x01\n\x0b\x45lementType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x08\n\x04TEXT\x10\x01\x12\x0f\n\x0bLEGEND_TEXT\x10\x02\x12\n\n\x06LEGEND\x10\x03\x12\t\n\x05HAXIS\x10\x04\x12\t\n\x05VAXIS\x10\x05\x12\n\n\x06\x43OLUMN\x10\x06\x12\x07\n\x03\x42\x41R\x10\x07\x12\t\n\x05TITLE\x10\x08\x12\x0f\n\x0b\x46RONT_TEXTS\x10\t\"\x8f\x01\n\x0c\x45lementReply\x12\x37\n\x06status\x18\x01 \x01(\x0b\x32\'.com.abcft.pdfextract.rpc.ocr.RpcStatus\x12\x37\n\x08\x65lements\x18\x02 \x03(\x0b\x32%.com.abcft.pdfextract.rpc.ocr.Element\x12\r\n\x05image\x18\x03 \x01(\x0c\x32\xc7\x02\n\nOcrService\x12g\n\rdetectElement\x12(.com.abcft.pdfextract.rpc.ocr.OcrRequest\x1a*.com.abcft.pdfextract.rpc.ocr.ElementReply\"\x00\x12\x62\n\x0bpredictText\x12(.com.abcft.pdfextract.rpc.ocr.OcrRequest\x1a\'.com.abcft.pdfextract.rpc.ocr.TextReply\"\x00\x12l\n\x10\x62\x61tchPredictText\x12-.com.abcft.pdfextract.rpc.ocr.BatchOcrRequest\x1a\'.com.abcft.pdfextract.rpc.ocr.TextReply\"\x00\x62\x06proto3')
)



_ELEMENT_ELEMENTTYPE = _descriptor.EnumDescriptor(
  name='ElementType',
  full_name='com.abcft.pdfextract.rpc.ocr.Element.ElementType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEXT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LEGEND_TEXT', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LEGEND', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HAXIS', index=4, number=4,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VAXIS', index=5, number=5,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COLUMN', index=6, number=6,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BAR', index=7, number=7,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TITLE', index=8, number=8,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FRONT_TEXTS', index=9, number=9,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=773,
  serialized_end=909,
)
_sym_db.RegisterEnumDescriptor(_ELEMENT_ELEMENTTYPE)


_RPCSTATUS = _descriptor.Descriptor(
  name='RpcStatus',
  full_name='com.abcft.pdfextract.rpc.ocr.RpcStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='success', full_name='com.abcft.pdfextract.rpc.ocr.RpcStatus.success', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error_code', full_name='com.abcft.pdfextract.rpc.ocr.RpcStatus.error_code', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error_msg', full_name='com.abcft.pdfextract.rpc.ocr.RpcStatus.error_msg', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=43,
  serialized_end=110,
)


_OCRREQUEST_PARAMSENTRY = _descriptor.Descriptor(
  name='ParamsEntry',
  full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.ParamsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.ParamsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.ParamsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=324,
  serialized_end=369,
)

_OCRREQUEST = _descriptor.Descriptor(
  name='OcrRequest',
  full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='url', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='params', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.params', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_name', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.image_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='need_denoise', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.need_denoise', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='need_denoise_image', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.need_denoise_image', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='need_parse_title', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.need_parse_title', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_OCRREQUEST_PARAMSENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='image', full_name='com.abcft.pdfextract.rpc.ocr.OcrRequest.image',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=113,
  serialized_end=378,
)


_BATCHOCRREQUEST = _descriptor.Descriptor(
  name='BatchOcrRequest',
  full_name='com.abcft.pdfextract.rpc.ocr.BatchOcrRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='images', full_name='com.abcft.pdfextract.rpc.ocr.BatchOcrRequest.images', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=380,
  serialized_end=455,
)


_TEXTREPLY = _descriptor.Descriptor(
  name='TextReply',
  full_name='com.abcft.pdfextract.rpc.ocr.TextReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='com.abcft.pdfextract.rpc.ocr.TextReply.status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='texts', full_name='com.abcft.pdfextract.rpc.ocr.TextReply.texts', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=457,
  serialized_end=540,
)


_BOX = _descriptor.Descriptor(
  name='Box',
  full_name='com.abcft.pdfextract.rpc.ocr.Box',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='center_x', full_name='com.abcft.pdfextract.rpc.ocr.Box.center_x', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='center_y', full_name='com.abcft.pdfextract.rpc.ocr.Box.center_y', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='com.abcft.pdfextract.rpc.ocr.Box.width', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='com.abcft.pdfextract.rpc.ocr.Box.height', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='rotate', full_name='com.abcft.pdfextract.rpc.ocr.Box.rotate', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=542,
  serialized_end=630,
)


_ELEMENT = _descriptor.Descriptor(
  name='Element',
  full_name='com.abcft.pdfextract.rpc.ocr.Element',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='com.abcft.pdfextract.rpc.ocr.Element.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bbox', full_name='com.abcft.pdfextract.rpc.ocr.Element.bbox', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='text', full_name='com.abcft.pdfextract.rpc.ocr.Element.text', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ELEMENT_ELEMENTTYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=633,
  serialized_end=909,
)


_ELEMENTREPLY = _descriptor.Descriptor(
  name='ElementReply',
  full_name='com.abcft.pdfextract.rpc.ocr.ElementReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='com.abcft.pdfextract.rpc.ocr.ElementReply.status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='elements', full_name='com.abcft.pdfextract.rpc.ocr.ElementReply.elements', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image', full_name='com.abcft.pdfextract.rpc.ocr.ElementReply.image', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=912,
  serialized_end=1055,
)

_OCRREQUEST_PARAMSENTRY.containing_type = _OCRREQUEST
_OCRREQUEST.fields_by_name['params'].message_type = _OCRREQUEST_PARAMSENTRY
_OCRREQUEST.oneofs_by_name['image'].fields.append(
  _OCRREQUEST.fields_by_name['data'])
_OCRREQUEST.fields_by_name['data'].containing_oneof = _OCRREQUEST.oneofs_by_name['image']
_OCRREQUEST.oneofs_by_name['image'].fields.append(
  _OCRREQUEST.fields_by_name['url'])
_OCRREQUEST.fields_by_name['url'].containing_oneof = _OCRREQUEST.oneofs_by_name['image']
_BATCHOCRREQUEST.fields_by_name['images'].message_type = _OCRREQUEST
_TEXTREPLY.fields_by_name['status'].message_type = _RPCSTATUS
_ELEMENT.fields_by_name['type'].enum_type = _ELEMENT_ELEMENTTYPE
_ELEMENT.fields_by_name['bbox'].message_type = _BOX
_ELEMENT_ELEMENTTYPE.containing_type = _ELEMENT
_ELEMENTREPLY.fields_by_name['status'].message_type = _RPCSTATUS
_ELEMENTREPLY.fields_by_name['elements'].message_type = _ELEMENT
DESCRIPTOR.message_types_by_name['RpcStatus'] = _RPCSTATUS
DESCRIPTOR.message_types_by_name['OcrRequest'] = _OCRREQUEST
DESCRIPTOR.message_types_by_name['BatchOcrRequest'] = _BATCHOCRREQUEST
DESCRIPTOR.message_types_by_name['TextReply'] = _TEXTREPLY
DESCRIPTOR.message_types_by_name['Box'] = _BOX
DESCRIPTOR.message_types_by_name['Element'] = _ELEMENT
DESCRIPTOR.message_types_by_name['ElementReply'] = _ELEMENTREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RpcStatus = _reflection.GeneratedProtocolMessageType('RpcStatus', (_message.Message,), dict(
  DESCRIPTOR = _RPCSTATUS,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.RpcStatus)
  ))
_sym_db.RegisterMessage(RpcStatus)

OcrRequest = _reflection.GeneratedProtocolMessageType('OcrRequest', (_message.Message,), dict(

  ParamsEntry = _reflection.GeneratedProtocolMessageType('ParamsEntry', (_message.Message,), dict(
    DESCRIPTOR = _OCRREQUEST_PARAMSENTRY,
    __module__ = 'ocr_pb2'
    # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.OcrRequest.ParamsEntry)
    ))
  ,
  DESCRIPTOR = _OCRREQUEST,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.OcrRequest)
  ))
_sym_db.RegisterMessage(OcrRequest)
_sym_db.RegisterMessage(OcrRequest.ParamsEntry)

BatchOcrRequest = _reflection.GeneratedProtocolMessageType('BatchOcrRequest', (_message.Message,), dict(
  DESCRIPTOR = _BATCHOCRREQUEST,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.BatchOcrRequest)
  ))
_sym_db.RegisterMessage(BatchOcrRequest)

TextReply = _reflection.GeneratedProtocolMessageType('TextReply', (_message.Message,), dict(
  DESCRIPTOR = _TEXTREPLY,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.TextReply)
  ))
_sym_db.RegisterMessage(TextReply)

Box = _reflection.GeneratedProtocolMessageType('Box', (_message.Message,), dict(
  DESCRIPTOR = _BOX,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.Box)
  ))
_sym_db.RegisterMessage(Box)

Element = _reflection.GeneratedProtocolMessageType('Element', (_message.Message,), dict(
  DESCRIPTOR = _ELEMENT,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.Element)
  ))
_sym_db.RegisterMessage(Element)

ElementReply = _reflection.GeneratedProtocolMessageType('ElementReply', (_message.Message,), dict(
  DESCRIPTOR = _ELEMENTREPLY,
  __module__ = 'ocr_pb2'
  # @@protoc_insertion_point(class_scope:com.abcft.pdfextract.rpc.ocr.ElementReply)
  ))
_sym_db.RegisterMessage(ElementReply)


_OCRREQUEST_PARAMSENTRY.has_options = True
_OCRREQUEST_PARAMSENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities


  class OcrServiceStub(object):
    # missing associated documentation comment in .proto file
    pass

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.detectElement = channel.unary_unary(
          '/com.abcft.pdfextract.rpc.ocr.OcrService/detectElement',
          request_serializer=OcrRequest.SerializeToString,
          response_deserializer=ElementReply.FromString,
          )
      self.predictText = channel.unary_unary(
          '/com.abcft.pdfextract.rpc.ocr.OcrService/predictText',
          request_serializer=OcrRequest.SerializeToString,
          response_deserializer=TextReply.FromString,
          )
      self.batchPredictText = channel.unary_unary(
          '/com.abcft.pdfextract.rpc.ocr.OcrService/batchPredictText',
          request_serializer=BatchOcrRequest.SerializeToString,
          response_deserializer=TextReply.FromString,
          )


  class OcrServiceServicer(object):
    # missing associated documentation comment in .proto file
    pass

    def detectElement(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def predictText(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def batchPredictText(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_OcrServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'detectElement': grpc.unary_unary_rpc_method_handler(
            servicer.detectElement,
            request_deserializer=OcrRequest.FromString,
            response_serializer=ElementReply.SerializeToString,
        ),
        'predictText': grpc.unary_unary_rpc_method_handler(
            servicer.predictText,
            request_deserializer=OcrRequest.FromString,
            response_serializer=TextReply.SerializeToString,
        ),
        'batchPredictText': grpc.unary_unary_rpc_method_handler(
            servicer.batchPredictText,
            request_deserializer=BatchOcrRequest.FromString,
            response_serializer=TextReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'com.abcft.pdfextract.rpc.ocr.OcrService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaOcrServiceServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    # missing associated documentation comment in .proto file
    pass
    def detectElement(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def predictText(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def batchPredictText(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaOcrServiceStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    # missing associated documentation comment in .proto file
    pass
    def detectElement(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    detectElement.future = None
    def predictText(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    predictText.future = None
    def batchPredictText(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    batchPredictText.future = None


  def beta_create_OcrService_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'batchPredictText'): BatchOcrRequest.FromString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'detectElement'): OcrRequest.FromString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'predictText'): OcrRequest.FromString,
    }
    response_serializers = {
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'batchPredictText'): TextReply.SerializeToString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'detectElement'): ElementReply.SerializeToString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'predictText'): TextReply.SerializeToString,
    }
    method_implementations = {
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'batchPredictText'): face_utilities.unary_unary_inline(servicer.batchPredictText),
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'detectElement'): face_utilities.unary_unary_inline(servicer.detectElement),
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'predictText'): face_utilities.unary_unary_inline(servicer.predictText),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_OcrService_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'batchPredictText'): BatchOcrRequest.SerializeToString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'detectElement'): OcrRequest.SerializeToString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'predictText'): OcrRequest.SerializeToString,
    }
    response_deserializers = {
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'batchPredictText'): TextReply.FromString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'detectElement'): ElementReply.FromString,
      ('com.abcft.pdfextract.rpc.ocr.OcrService', 'predictText'): TextReply.FromString,
    }
    cardinalities = {
      'batchPredictText': cardinality.Cardinality.UNARY_UNARY,
      'detectElement': cardinality.Cardinality.UNARY_UNARY,
      'predictText': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'com.abcft.pdfextract.rpc.ocr.OcrService', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)
