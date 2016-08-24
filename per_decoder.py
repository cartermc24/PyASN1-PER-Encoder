""" PyASN1PERDecoder

This module will encode PyASN1 objects into the unaligned PER format.
The specification for the PER encoding format is online at:
    https://www.itu.int/ITU-T/studygroups/com17/languages/X.691-0207.pdf

This class performs a recursive decoding where each sequence or choice steps into another
instance of _decode_sequence.  Primitive objects are directly decoded into PyASN.1 and appended to
decoded_result as binary.  If an object is empty or null-deterministic as specified by X.691 it is
not decoded and ignored (i.e. returns None).  When a sequence or choice is encountered, a bitmap
of the structure is generated, a new PyASN.1 object is populated and the object is stepped into.
After the stepped sequence or choice is decoded, the decoding continues where it left off at the
parent structure.  Once decoding is done, the populated PyASN.1 object is returned.

PyASN1 objects have descriptions accessed using repr or other built-in functions to obtain their
values, constraints, and more.  This class uses those functions to avoid having to change how
PyASN1 works (as deep parameters about the objects are hidden).  Examples of this is using repr
and __bases__.

How to Use:
    decoded_pyasn1_object = per_encoder.per_unaligned_decoder(unaligned PER encoded bytes)

Supported ASN.1 Objects:
    - SEQUENCE
    - CHOICE
    - BITSTRING
    - INTEGER
    - BOOLEAN
    - OCTET_STRING
The decoder will raise an exception if an unsupported object is encountered
"""

import logging
from pyasn1.type import univ, base
from typing import Tuple, Union
from importlib import import_module
from .per_encoder import _bit_str_pad, _extract_value_range_constraint, \
                        _extract_value_size_constraint


def per_unaligned_decode(base_item: Union[univ.Sequence, univ.Choice], data: bytes,
                         ref_class: str) -> base.Asn1Item:
    """ Performs a PER Unaligned decode of data based on the base_item structure

    Args:
        base_item: The ASN.1 base choice or sequence that defines the encoded data
        data: The PER Unaligned encoded data in bytes
        ref_class: A string that references the class where the ASN.1 objects are defined

    Returns:
        The base_item is returned with the data fields populated

    """
    logging.info('[PyASN1PERDecoder]: Decode started')
    data_bin = ''
    for i in data:
        data_bin += _bit_str_pad(str(bin(i))[2:], 8, '0')

    processed, decoded_item = _decode_sequence(base_item, data_bin, ref_class)

    if processed != len(data_bin):
        unprocessed_data = False
        for i in range(processed, len(data_bin)):
            if data_bin[i] == '1':
                unprocessed_data = True
                break
        if unprocessed_data:
            raise RuntimeError('[PyASN1PERDecoder][ERROR]: Did not fully decode data, length of '
                               'input data [{}], decoded [{}]'.format(len(data_bin), processed))

    logging.info('[PyASN1PERDecoder]: Decode completed')
    return decoded_item


def _decode_sequence(base_item: Union[univ.Sequence, univ.Choice], data: str,
                     ref_class: str) -> Tuple[int, univ.Sequence]:
    """ Decodes an ASN.1 SEQUENCE or CHOICE object.  Reads the PyASN.1 definition and reads the
        metadata for each item and populates them into lists that define the structure.  The items
        are then decoded in order recursively decoding sequences as they are encountered.

    Args:
        base_item: The PyASN.1 object that defines the structure (can be empty)
        data: The PER unaligned encoded binary string
        ref_class: The module that contains all the PyASN.1 objects that are defined in the
            sequence

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    if str(base_item.__class__.__bases__).find('pyasn1.type.univ.Choice') != -1:
        choice = True
    else:
        choice = False

    parse_index = 0
    sequence_description = str(base_item.getComponentType())
    required = []
    field_type = []
    field_name = []
    description = []
    objects = []

    id_length = len('NamedTypes')
    if sequence_description[:id_length] != 'NamedTypes':
        raise ValueError('[PyASN1PERDecoder][ERROR]: SEQUENCE/CHOICE description invalid')

    optional_components = False
    nest = 0
    # Nest 1 -> NamedTypes
    # Nest 2 -> NamedType
    # Nest 3 -> Type
    # Nest 4 -> Constraint
    buffer = ''
    for i in range(0, len(sequence_description)):
        if sequence_description[i] == '(':
            if nest == 1:
                if buffer.find('OptionalNamedType') != -1:
                    required.append('O')
                    optional_components = True
                elif buffer.find('NamedType') != -1:
                    required.append('R')
                else:
                    raise ValueError('[PyASN1PERDecoder][ERROR]: Sequence contains an unknown '
                                     'type, cannot encode unknown types')
                field_name.append(sequence_description[i+2:sequence_description
                                  .find('\'', i+2)])
                field_type.append(sequence_description[sequence_description.find(
                    '\'', i+2)+3:sequence_description.find('(', i+3)])
                buffer = ''

                target = ''
                internal_range = 1

                # Are there properties to capture?
                o_pos = sequence_description.find('(', i+1)
                c_pos = sequence_description.find(')', i+1)
                if abs(c_pos-o_pos) != 1:
                    # Capture segment from item
                    for k in range(sequence_description.find('=', i+1)+1,
                                   len(sequence_description)):
                        if sequence_description[k] == '(':
                            internal_range += 1
                        elif sequence_description[k] == ')':
                            internal_range -= 1
                        if internal_range == 0:
                            break
                        target += sequence_description[k]
                    description.append(target)
                else:
                    description.append('None')
            nest += 1
        if sequence_description[i] == ')':
            nest -= 1
            buffer = ''
        buffer += sequence_description[i]

    for j in range(0, len(field_type)):
        if _is_complex_type(field_type[j]):
            module = import_module(ref_class)
            i_class = getattr(module, field_type[j])
            instance = i_class()
            objects.append(instance)
        else:
            objects.append(description[j])

    if choice:
        max_bits = len(str(bin(len(field_name) - 1))[2:])
        chosen_index = int(data[parse_index:max_bits], 2)
        parse_index += max_bits
        logging.debug('Identified as CHOICE, chosen item as [{}]'.format(field_name[chosen_index]))
        logging.debug('Decoding [{}] as [{}]'.format(field_name[chosen_index],
                                                     field_type[chosen_index]))
        parser_offset, decoded_item = _identify_decode_obj(objects[chosen_index], field_type[
                                                           chosen_index], data[parse_index:],
                                                           ref_class)
        parse_index += parser_offset
        logging.debug('Successfully decoded [{}] with value [{}]'.format(field_name[chosen_index],
                                                                         decoded_item))
        base_item.setComponentByName(field_name[chosen_index], decoded_item)
    else:
        opt_pos = -1  # -1 due to data index starting at 0
        optional_components_count = 0
        if optional_components:  # Account for bitmap if present
            for i in range(0, len(required)):
                if required[i] == 'O':
                    optional_components_count += 1
            logging.debug('OCD: [{}] optional items in  SEQUENCE'.format(optional_components_count
                                                                         ))
            parse_index += optional_components_count
        for i in range(0, len(required)):
            if required[i] == 'O':
                opt_pos += 1
                if data[opt_pos] == '0':
                    logging.debug('Item [{}] was optional and not included, skipping'.format(
                        field_name[i]))
                    continue

            logging.debug('Decoding [{}] as [{}]'.format(field_name[i], field_type[i]))
            parser_offset, decoded_item = _identify_decode_obj(objects[i], field_type[i],
                                                               data[parse_index:], ref_class)
            parse_index += parser_offset
            logging.debug('Successfully decoded [{}] with value [{}]'.format(field_name[i],
                                                                             decoded_item))
            base_item.setComponentByName(field_name[i], decoded_item)

    return parse_index, base_item


def _is_complex_type(item_type: str) -> bool:
    """ Checks to see if the type is a complex type (as primitives are decoded differently)

    Args:
        item_type: A string that has the type of the object

    Returns:
        True if complex, False if primitive

    """
    if item_type == 'Boolean' or item_type == 'BitString' or item_type == 'Integer':
        return False
    else:
        return True


def _identify_decode_obj(item: Union[base.Asn1Item, str], item_type: str, data: str,
                         ref_class: str) -> Tuple[int, base.Asn1Item]:
    """ Decodes a PER unaligned encoded ASN.1 item, identifies the type of the item and passes
    it to the appropriate decoder

    Args:
        item: The PyASN.1 item that describes the object if complex, or the string description
            of the item if a primitive
        item_type: The type of the item to decode
        data: The PER unaligned encoded binary string
        ref_class: The module that contains all the PyASN.1 objects that are defined in the
            sequence

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    class_base = str(item.__class__.__bases__)
    if class_base.find('pyasn1.type.univ.Sequence') != -1:
        return _decode_sequence(item, data, ref_class)
    elif item_type == 'Boolean':
        return _decode_boolean(data)
    elif class_base.find('pyasn1.type.univ.Enumerated') != -1:
        return _decode_enumeration(item, data)
    elif class_base.find('pyasn1.type.univ.OctetString') != -1:
        return _decode_octet_string(item, data)
    elif class_base.find('pyasn1.type.univ.Choice') != -1:
        return _decode_sequence(item, data, ref_class)
    elif item_type == 'BitString' != -1:
        return _decode_bitstring(item, data)
    elif item_type == 'Integer' != -1:
        return _decode_integer(item, data)
    else:
        raise ValueError('[PyASN1PERDecoder][ERROR]: Cannot identify type in sequence '
                         'with base defined by: {}\nRepresentation: {}'
                         '\nBase:{}'.format(class_base, repr(item), class_base))
    pass


def _decode_boolean(data: str) -> Tuple[int, base.Asn1Item]:
    """ Decodes an ASN.1 PER Encoded boolean object

    Args:
        data: The PER unaligned encoded binary string

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    if data[0] == '1':
        return 1, univ.Boolean(value=True)
    else:
        return 1, univ.Boolean(value=False)


def _decode_enumeration(item: univ.Enumerated, data: str) -> Tuple[int, base.Asn1Item]:
    """ Decodes an ASN.1 PER Encoded enumeration object

    Args:
        item: The enumerated base object that defines the encoded data
        data: The PER unaligned encoded binary string

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    values = str(item.getNamedValues())
    max_enum_index = -1
    for x in range(0, len(values)):
        if values[x] == '(':
            max_enum_index += 1

    if max_enum_index == 0 or max_enum_index == 1:
        return 0, item.clone(0)  # Empty/non-deterministic enum
    elif max_enum_index > 256:
        raise IndexError('[PyASN1PERDecoder][ERROR]: Enumerations larger than 256 items are '
                         'not supported')

    offset_bitfield_range = len(str(bin(max_enum_index - 1)[2:]))
    target_value = int(data[:offset_bitfield_range], 2)

    target = -2
    enum_name = ''
    for i in range(0, len(values)):
        if values[i] is '(':
            target += 1
        elif target is target_value:
            end_target = values.find('\'', i+1)
            enum_name = values[i+1:end_target]
            break

    if enum_name is '':
        raise RuntimeError('[PyASN1PERDecoder][ERROR]: Invalid value for enumeration given base')

    decoded_enum = item.clone(enum_name)
    return offset_bitfield_range, decoded_enum


def _decode_octet_string(item: univ.OctetString, data: str) -> Tuple[int, base.Asn1Item]:
    """ Decodes an ASN.1 PER Encoded octet-string object

    Args:
        item: The enumerated base object that defines the encoded data
        data: The PER unaligned encoded binary string

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    dec_constraint = _extract_value_size_constraint(str(item.getSubtypeSpec()))

    if dec_constraint == -1:
        # No constraint
        length = int(data[:8], 2)
        decoded_str = ''
        for i in range(0, length):
            decoded_str += chr(int(data[(8+i*8):(16+i*8)], 2))
        return 8*(length+1), item.clone(decoded_str)
    elif dec_constraint[1] == 0:
        return 0, item.clone()
    elif dec_constraint[0] == dec_constraint[1]:
        decoded_str = ''
        for i in range(0, dec_constraint[0]):
            decoded_str += chr(int(data[(8+i*8):(16+i*8)], 2))
        return 8*dec_constraint[0], item.clone(decoded_str)
    else:
        offset_bitfield_range = len(str(bin(dec_constraint[1]-dec_constraint[0])[2:]))
        length = int(data[:offset_bitfield_range], 2) + dec_constraint[0]
        decoded_str = ''
        for i in range(0, length):
            decoded_str += chr(int(data[(offset_bitfield_range+(i*8)):(
                (offset_bitfield_range+8)+i*8)], 2))
        return offset_bitfield_range+(8*length), item.clone(decoded_str)


def _decode_bitstring(description: str, data: str) -> Tuple[int, base.Asn1Item]:
    """ Decodes an ASN.1 PER Encoded bitstring object

    Args:
        description: The description string of the bitstring (containing constraints, etc)
        data: The PER unaligned encoded binary string

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    dec_constraint = _extract_value_size_constraint(description)

    if dec_constraint == -1:
        return 0, univ.BitString()  # No decoding required if length constrained to zero
    elif dec_constraint[0] == dec_constraint[1]:
        decoded_bitstring = '\'' + data[:dec_constraint[0]] + '\'B'
        return dec_constraint[0], univ.BitString(decoded_bitstring)
    else:  # if ub != lb
        offset_bitfield_range = len(str(bin(dec_constraint[1]-dec_constraint[0]))[2:])
        length = int(data[:offset_bitfield_range], 2)
        decoded_bitstring = '\'' + data[:length] + '\'B'
        return offset_bitfield_range + length, univ.BitString(decoded_bitstring)


def _decode_integer(description: str, data: str) -> Tuple[int, base.Asn1Item]:
    """ Decodes an ASN.1 PER Encoded integer object
        Note: the decoder currently does not support negative integers while the encoder does

    Args:
        description: The description string of the integer (containing constraints, etc)
        data: The PER unaligned encoded binary string

    Returns:
        A tuple containing the PyASN.1 object populated with the decoded data and the
        length of the data processed (used for error checking)

    """
    dec_constraint = _extract_value_range_constraint(description)
    # Determine if constrained or not
    if dec_constraint == -1:  # unconstrained
        length = int(data[:8], 2)
        decoded_integer = int(data[8:(length*8)+8], 2)
        return (length+1) * 8, univ.Integer(decoded_integer)
    else:  # constrained
        offset = dec_constraint[1] - dec_constraint[0]
        if offset == 1:
            return 0, univ.Integer(dec_constraint[0])  # No encoding required if range is only 1
        offset_bitfield_range = len(str(bin(offset))[2:])
        decode_int_offset = int(data[:offset_bitfield_range], 2)
        decoded_integer = dec_constraint[0] + decode_int_offset
        return offset_bitfield_range, univ.Integer(decoded_integer)
