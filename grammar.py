grammar = """
start: expr+

expr: classify | replace | load | save | say | detect | cutout | size | count | countinregion | in | if | segment | BOOL | EQUALITY | var | EOL | variable | comment | show | exit | help | train | compare | read | label | use | list | get | INT
classify: "Classify" "[" STRING ("," STRING)* "]"
var: variable "=" expr
replace: "Replace" "[" STRING "]"
use: "Use" "[" STRING "]"
load: "Load" "[" STRING "]" | "Load" "[" "]"
save: "Save" "[" STRING "]"
say: "Say" "[" "]"
size: "Size" "[" "]"
show: "Show" "[" "]"
cutout: "Cutout" "[" "]"
count: "Count" "[" "]"
compare: "Compare" "[" "]"
read: "Read" "[" "]"
exit: "Exit" "[" "]"
get: "Get" "[" INT "]" EOL
list: "[" ((STRING | INT | expr) "," | (STRING | INT | expr) )* "]" EOL
help: "Help" "[" STRING "]"
countinregion: "CountInRegion" "[" INT "," INT "," INT "," INT "]"
detect: "Detect" "[" STRING ("," STRING)* "]" | "Detect" "[" "]"
segment: "Segment" "[" STRING "]"
in: "IN" "[" STRING "]" EOL (INDENT expr+)*
if: "IF" "[" (expr+) "]" EOL (INDENT expr+)*
OPERAND: "+" | "-" | "*" | "/"
EQUALITY: "=="
train: "Train" "[" STRING "," STRING "]" | "Train" "[" STRING "]"
label: "Label" "[" STRING "," STRING ("," STRING )*  "]" 
variable: /[a-zA-Z]+/
comment: /#.*$/ (expr)*
EOL: "\\n"
INT: /-?\d+/
INDENT: "    "
BOOL: "True" | "False"
%import common.ESCAPED_STRING -> STRING
%import common.WS_INLINE
%ignore WS_INLINE
"""