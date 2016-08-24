""" PyASN1PEREncoder

This module will encode PyASN1 objects into the unaligned PER format.
The specification for the PER encoding format is online at:
    https://www.itu.int/ITU-T/studygroups/com17/languages/X.691-0207.pdf

This class performs a recursive encoding where each sequence or choice steps into another
instance of _encode_sequence.  Primitive objects are directly encoded into binary and appended to
encoded_result as binary.  If an object is empty or null-deterministic as specified by X.691 it is
not encoded an ignored (i.e. returns None).  When a sequence or choice is encountered, a bitmap
of the structure is generated, appended to encoded_result as binary and the object is stepped into.
After the stepped sequence or choice is encoded, the encoding continues where it left off at the
parent structure.  Once encoding is done, the binary is turned into ASCII values and returned.

PyASN1 objects have descriptions accessed using repr or other built-in functions to obtain their
values, constraints, and more.  This class uses those functions to avoid having to change how
PyASN1 works (as deep parameters about the objects are hidden).  Examples of this is using repr
and __bases__.

How to Use:
    encoded_bytes = per_encoder.per_unaligned_encode(PyASN1 SEQUENCE or CHOICE)

Supported ASN.1 Objects:
    - SEQUENCE
    - CHOICE
    - BITSTRING
    - INTEGER
    - BOOLEAN
    - OCTET_STRING
The encoder will raise an exception if an unsupported object is encountered
"""

import ast
import logging
from pyasn1.type import univ, base
from typing import Tuple, Union, Optional


def per_unaligned_encode(obj: base.Asn1Item) -> bytes:
    """ Encodes a PyASN1 data object using unaligned PER
    Args:
        obj: A PyASN1 SEQUENCE or CHOICE object

    Returns:
        A hex string containing the encoded data

    """
    logging.info('[PyASN1PEREncoder]: Starting Encode...')

    encoded_result = _encode_sequence(obj)

    if (len(encoded_result) % 8) is not 0:
        req = -(-len(encoded_result) // 8)  # Rounds up
        encoded_result = _bit_str_right_pad(encoded_result, req * 8, '0')

    # This statement iterates through each byte in the binary string and turns it into an
    # integer which is then fed into chr to encode it into ASCII - each byte is then appended
    # to the string
    encoded = ''.join(chr(int(encoded_result[i:i + 8], 2))
                      for i in range(0, len(encoded_result), 8))

    encoded_bytes = b''
    for x in range(0, len(encoded)):
        # Re-encode the results from the encoder into bytes (in hex format)
        # the type error thrown by PyCharm is normal
        encoded_bytes += ord(encoded[x]).to_bytes(1, byteorder='big')

    logging.info('[PyASN1PEREncoder]: Encoding Complete')
    logging.debug('[PyASN1PEREncoder]: PER Encoded value:\n[' + str(encoded_bytes) + ']')
    return encoded_bytes


def _encode_sequence(sequence_obj: Union[univ.Choice, univ.Sequence]) -> str:
    """ Encodes a sequence or choice stepping into nested sequences or choices
    Args:
        sequence_obj: A PyASN1 SEQUENCE or CHOICE object

    Returns:
        String containing binary encoded value

    """
    if str(sequence_obj.__class__.__bases__).find('pyasn1.type.univ.Choice') != -1:
        choice = True
    else:
        choice = False

    encoded_result = ''
    sequence_description = str(sequence_obj.getComponentType())
    required = []
    field_type = []
    field_name = []
    objects = []
    description = []

    id_length = len('NamedTypes')
    if sequence_description[:id_length] != 'NamedTypes':
        raise ValueError('[PyASN1PEREncoder][ERROR]: SEQUENCE/CHOICE description invalid')

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
                elif buffer.find('NamedType') != -1:
                    required.append('R')
                else:
                    raise ValueError('[PyASN1PEREncoder][ERROR]: Sequence contains an unknown '
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

    present_items = 0
    present_index = -1

    bitmap = ''

    # Generate bitmap and verify that the required fields are included
    for j in range(0, len(required)):
        objects.append(sequence_obj.getComponentByName(field_name[j]))
        if objects[j] is not None:
            present_items += 1
            present_index = j

        if objects[j] is None and required[j] == 'R':
            if not choice:
                logging.error('[PyASN1PEREncoder]: Required object in sequence not '
                              'present - Missing [' + field_name[j] + '], which is required')
        elif objects[j] is None and required[j] == 'O':
            bitmap += '0'
        elif objects[j] is not None and required[j] == 'O':
            bitmap += '1'

    logging.debug('[PyASN1PEREncoder]: Bitmap for sequence: [' + bitmap + ']')

    if not choice:
        encoded_result += bitmap
    elif present_items == 1:
        logging.debug('[PyASN1PEREncoder]: Choice verification passed')
        max_bits = len(str(bin(len(field_name) - 1))[2:])
        choice_header = _bit_str_pad(bin(present_index)[2:], max_bits, '0')
        encoded_result += choice_header
        logging.debug('[PyASN1PEREncoder]: Choice header: [' + choice_header + ']')
    else:
        raise IndexError('[PyASN1PEREncoder][ERROR]: Selected multiple items for choice - '
                         'invalid, cannot encode')

    logging.debug('[PyASN1PEREncoder]:')

    for x in range(0, len(required)):
        if objects[x] is not None:
            stat = 'Present'
        else:
            stat = 'Missing'
        logging.debug('\t[SEQUENCE ITEM PARSED]: Name:[' + field_name[x] + '] Type:[' +
                      field_type[x] + '] Required:[' + required[x] + '] Present:[' + stat +
                      ']')

    logging.debug('[PyASN1PEREncoder]: Starting sequence encode')
    # Main encoding logic, step through each item
    for y in range(len(required)):
        if objects[y] is not None:
            logging.debug('[PyASN1PEREncoder]: Encoding [' + field_name[y] + '] as [' +
                          field_type[y] + ']')
            result = _check_encode_primitive_type(field_type[y], objects[y])
            if result is not None:
                logging.debug('[PyASN1PEREncoder]: [' + field_name[y] + '] '
                              'encoded as [' + result + ']')
                encoded_result += result
            else:
                logging.debug('[PyASN1PEREncoder]: [' + field_name[y] + '] not a '
                              'primitive - stepping into object')
                encoded_result += _check_encode_complex_type(objects[y])
    return encoded_result


def _encode_octet(obj: univ.OctetString) -> str:
    """ Encodes a STRING OCTET type
    Args:
        obj: A PyASN1 STRING_OCTET type

    Returns:
        A string containing the encoded binary

    """
    dec_constraint = _extract_value_size_constraint(str(obj.getSubtypeSpec()))
    oct_str = str(obj)
    if dec_constraint == -1:
        # No constraint
        length = len(str(obj))
        len_det = _bit_str_pad(bin(length)[2:], 8, '0')
        return len_det + ''.join([_bit_str_pad(bin(ord(oct_str[x]))[2:], 8, '0')
                                  for x in range(len(oct_str))])
    if dec_constraint[1] == 0:
        return None
    elif dec_constraint[0] == dec_constraint[1]:
        if len(obj) == dec_constraint[0]:
            return ''.join([_bit_str_pad(bin(ord(oct_str[x]))[2:], 8, '0') for x in range(
                len(oct_str))])
        else:
            raise ValueError('[PyASN1PEREncoder][ERROR]: Octet string does not fit the '
                             'constraints - cannot encode [len != const_len]')
    elif len(oct_str) < dec_constraint[0] or len(oct_str) > dec_constraint[1]:
        raise ValueError('[PyASN1PEREncoder][ERROR]: Octet string does not fit the '
                         'constraints - cannot encode ![lb < x < ub]')
    else:
        offset_bitfield_range = len(str(bin(dec_constraint[1]-dec_constraint[0])[2:]))
        len_det = _bit_str_pad(bin(len(oct_str)-dec_constraint[0])[2:], offset_bitfield_range, '0')
        return len_det + ''.join([_bit_str_pad(bin(ord(oct_str[x]))[2:], 8, '0')
                                  for x in range(len(oct_str))])


def _check_encode_primitive_type(field_type: str, obj: base.Asn1Item) -> Optional[str]:
    """ Encodes PyASN1 primitives
    Args:
        field_type: A string containing the type of object
        obj: The object to encode

    Returns:
        The binary encoded value as a string, none if the object cannot be encoded

    """
    if field_type == 'Integer':
        return _encode_int(obj)
    elif field_type == 'Boolean':
        return _encode_bool(obj)
    elif field_type == 'BitString':
        return _encode_bitstring(obj)
    else:
        return None


def _check_encode_complex_type(obj: base.Asn1Item) -> str:
    """ Encodes PyASN1 complex objects
    Args:
        obj: The object to encode

    Returns:
        The binary encoded value as a string
    """
    class_base = str(obj.__class__.__bases__)
    if class_base.find('pyasn1.type.univ.Sequence') != -1:
        logging.debug('[PyASN1PEREncoder]: Identified as nested sequence, '
                      'stepping in')
        return _encode_sequence(obj)
    elif class_base.find('pyasn1.type.univ.Enumerated') != -1:
        logging.debug('[PyASN1PEREncoder]: Identified as enumeration, encoding as enumeration')
        return _encode_enumeration(obj)
    elif class_base.find('pyasn1.type.univ.OctetString') != -1:
        logging.debug('[PyASN1PEREncoder]: Identified as based in OctetString, encoding as '
                      'OctetString')
        return _encode_octet(obj)
    elif class_base.find('pyasn1.type.univ.Choice') != -1:
        logging.debug('[PyASN1PEREncoder]: Identified as Choice, encoding as '
                      'Choice')
        return _encode_sequence(obj)
    else:
        raise ValueError('[PyASN1PEREncoder][ERROR]: Cannot identify complex type in sequence '
                         'with base defined by: ' + class_base + '\nRepresentation: ' +
                         repr(obj))


def _encode_bool(obj: univ.Boolean) -> str:
    """ Encodes a PyASN1 boolean type
    Args:
        obj: The PyASN1 boolean object

    Returns:
        '1' if true, '0' if false

    """
    if obj:
        return '1'
    else:
        return '0'


def _encode_enumeration(obj: univ.Enumerated) -> str:
    """ Encodes a PyASN1 enumeration type
    This method iterates over the PyASN1 definition string to obtain the enumeration values
    and other attributes about the enumeration.
    Args:
        obj: The PyASN1 enumeration object

    Returns:
        A string with the encoded value

    """
    values = str(obj.getNamedValues())
    max_enum_index = -1
    for x in range(0, len(values)):
        if values[x] == '(':
            max_enum_index += 1

    enum_index_marker = values.find(obj.prettyPrint())
    enum_index = -2
    for i in range(0, enum_index_marker):
        if values[i] is '(':
            enum_index += 1

    if max_enum_index == 0 or max_enum_index == 1:
        return ''  # Empty/non-deterministic enum
    elif max_enum_index > 256:
        raise IndexError('[PyASN1PEREncoder][ERROR]: Enumeration larger than 256 items, '
                         'not supported')
    offset_bitfield_range = len(str(bin(max_enum_index - 1)[2:]))
    bin_str = _bit_str_pad(str(bin(enum_index))[2:], offset_bitfield_range, '0')
    logging.debug('[PyASN1PEREncoder]: Enumeration value [' + bin_str + ']')
    return bin_str


def _encode_bitstring(obj: univ.BitString) -> str:
    """ Encodes a PyASN1 bitstring object
    Args:
        obj: The PyASN1 bitstring object

    Returns:
        A string with the encoded value

    """
    if str.find(repr(obj), 'BitString') == -1:
        raise ValueError('[PyASN1PEREncoder][ERROR]: Value sent to BitString encoder not a '
                         'BitString - Expected [BitString] got: [' + str(type(obj)) + ']')
    dec_constraint = _extract_value_size_constraint(repr(obj))
    obj = str(obj)
    obj = obj.replace(',', '')
    obj = obj.replace(' ', '')
    obj = obj.replace('(', '')
    obj = obj.replace(')', '')

    if dec_constraint == -1:
        return ''  # No encoding required if length constrained to zero

    if len(obj) < dec_constraint[0] or len(obj) > dec_constraint[1]:
        raise OverflowError('[PyASN1PEREncoder][ERROR]: Value outside of constraint for field,'
                            ' encoding cannot continue\nExpected [' + dec_constraint[0] +
                            '<val<' + dec_constraint[1] + '] got [' + str(len(obj)) + ']')

    if dec_constraint[0] == dec_constraint[1]:
        return obj  # If ub == lb, just append the BitString
    else:  # if ub != lb
        offset_bitfield_range = len(str(bin(dec_constraint[1]-dec_constraint[0]))[2:])
        str_len = len(str(obj))
        size_str = _bit_str_pad(str(bin(str_len)), offset_bitfield_range, '0')
        return size_str + obj


def _encode_int(obj: univ.Integer) -> str:
    """ Encodes a PyASN1 integer object


    Args:
        obj: The PyASN1 integer object

    Returns:
        A string with the encoded value

    """
    if str.find(repr(obj), 'Integer') == -1:
        raise ValueError('[PyASN1PEREncoder][ERROR]: Value sent to integer encoder not an '
                         'PyASN1 integer - Expected [INTEGER] got: [' + str(type(obj)) + ']')
    dec_constraint = _extract_value_range_constraint(repr(obj))
    # Determine if constrained or not
    if dec_constraint == -1:  # unconstrained
        return _int_to_bin_with_oct_len(obj)
    else:  # constrained
        if obj < dec_constraint[0] or obj > dec_constraint[1]:
            raise ValueError('[PyASN1PEREncoder][ERROR]: Value outside of constraint for '
                             'field, encoding cannot continue - Expected [' +
                             str(dec_constraint[0]) + '<val<' + str(dec_constraint[1]) + '] '
                             'got [' + obj + ']')
        offset = dec_constraint[1] - dec_constraint[0]
        if offset == 1:
            return ''  # No encoding required if range is only 1
        offset_bitfield_range = len(str(bin(offset))[2:])
        len_det = int(obj) - dec_constraint[0]
        bin_str = _bit_str_pad(str(bin(len_det))[2:], offset_bitfield_range, '0')
        return bin_str


def _int_to_bin_with_oct_len(obj: int='0') -> str:
    """ Converts an integer to binary with an octal length (factor of 8)
    Args:
        obj: The integer to convert

    Returns:
        A string containing an octal length binary representation of obj

    """
    if obj >= 0:
        bin_val = str(bin(obj))[2:]
        req_len = int(-(-len(bin_val)//8))  # this rounds up after division
        bin_val = _bit_str_pad(bin_val, req_len * 8, '0')
    else:
        req_len, bin_val = _convert_twos_complement(obj)
    # Generate length string
    length_str = _bit_str_pad(str(bin(req_len))[2:], 8, '0')
    return length_str + bin_val


def _convert_twos_complement(obj: int) -> Tuple[int, str]:
    """ Converts a negative integer to a twos complement binary string to the nearest octet
    Args:
        obj: Negative integer

    Returns:
        The required length in octets of the binary two's complement representation and
        a string of octal length of the two's complement representation of the integer

    """
    if obj < -32768:
        raise OverflowError('[PyASN1PEREncoder][ERROR]: Value too small to encode, hard limit '
                            'set in per_encoder.py')

    x = 0
    for x in range(7, 32, 8):
        if -1*obj < pow(2, x):
            break
    total_remainder = pow(2, x) + obj
    bin_str = ''
    for y in range(x, -1, -1):
        if pow(2, y) <= total_remainder:
            bin_str += '1'
            total_remainder -= pow(2, y)
        else:
            bin_str += '0'
    bin_str = '1' + bin_str[1:]
    return int((x + 1)/8), bin_str


def _bit_str_pad(base_str: str, req_len: int, pad_type: str='0') -> str:
    """ Pads the left side of a string with pad_type
    Args:
        base_str: The string to add padding to
        req_len: The total required length of the string
        pad_type: What to pad the string with

    Returns:
        A left padded string based on the required length

    """
    for x in range(len(base_str), req_len):
        base_str = pad_type + base_str
    return base_str


def _bit_str_right_pad(base_str: str, req_len: int, pad_type: str='0') -> str:
    """ Pads the right side of a string with pad_type
    Args:
        base_str: The string to add padding to
        req_len: The total required length of the string
        pad_type: What to pad the string with

    Returns:
        A left padded string based on the required length

    """
    for x in range(len(base_str), req_len):
        base_str += pad_type
    return base_str


def _extract_value_range_constraint(description: str) -> Tuple[int, int]:
    """ Extracts the constrains from value range based PyASN1 objects
    Args:
        description: The description of a PyASN1 range based object

    Returns:
        A tuple of the minimum and maximum value ranges, -1 if no constraints or if the
        description is formatted incorrectly

    """
    index = str.find(description, 'ValueRangeConstraint')
    if index == -1:
        return -1
    index += 20
    tup = ''
    for x in range(index, len(description)):
        tup += description[x]
        if description[x] == ')':
            break
    return ast.literal_eval(tup)


def _extract_value_size_constraint(description: str) -> Tuple[int, int]:
    """ Extracts the constrains from value size based PyASN1 objects
    Args:
        description: The description of a PyASN1 size based object

    Returns:
        A tuple of the minimum and maximum sizes, -1 if no constraints or if the
        description is formatted incorrectly

    """
    index = str.find(description, 'ValueSizeConstraint')
    if index == -1:
        return -1
    index += 19
    tup = ''
    for x in range(index, len(description)):
        tup += description[x]
        if description[x] == ')':
            break
    return ast.literal_eval(tup)
