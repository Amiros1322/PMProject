<% let dtypeToCTypeMap = { 
	"u1": "uint8_t" , 
	"u2": "uint16_t", 
	"u4": "uint32_t", 
	"u8": "uint64_t", 
	"s1": "int8_t", 
	"s2": "int16_t", 
	"s4": "int32_t", 
	"s8": "int64_t", 
	"f4": "float", 
	"f8": "double"
}; -%>
<% for (const dtype in dtypeToCTypeMap){ -%>
<% let ctype = dtypeToCTypeMap[dtype]; -%>
/* <%=dtype%> setters */
void set_<%=dtype%>(size_t index, <%=ctype%> value);

void set_<%=dtype%>_s(size_t index, <%=ctype%> value);

void set_<%=dtype%>(size_t r, size_t c, <%=ctype%> value);

void set_<%=dtype%>_s(size_t r, size_t c, <%=ctype%> value);

<% }; -%> 