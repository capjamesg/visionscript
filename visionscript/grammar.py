grammar = """
start: (expr)*

expr: (if | in | train | label | detect | countinregion | help | list | get | exit | read | compare | count | cutout | show | size | caption | say | save | load | use | replace | var | classify | segment | comment | contains | if | else | end | tutorial | make | run | isita | find | describe) (EOL | EOF | " ")
classify: "Classify" "[" STRING ("," STRING)* "]"
var: variable "=" expr
replace: "Replace" "[" STRING "]"
use: "Use" "[" STRING "]"
load: "Load" "[" STRING "]" | "Load" "[" "]"
save: "Save" "[" STRING "]"
say: "Say" "[" "]"
describe: "Describe" "[" "]"
isita: "Is it a " (("," STRING)* | ("or" STRING)*)? EOL
find: "Find" "[" STRING "]"
args: ((STRING | INT | expr) ("," (STRING | INT | expr))*) | (STRING | INT | expr)?
make: "Make" STRING "[" args "]" EOL (INDENT expr+)*
caption: "Caption" "[" "]"
size: "Size" "[" "]"
run: "Run" "[" STRING "]"
show: "Show" "[" "]"
cutout: "Cutout" "[" "]"
count: "Count" "[" "]"
contains: "Contains" "[" STRING "]"
compare: "Compare" "[" "]"
read: "Read" "[" "]"
exit: "Exit" "[" "]"
get: "Get" "[" INT "]" EOL
list: "[" ((STRING | INT | expr) "," | (STRING | INT | expr) )* "]" EOL
help: "Help" "[" STRING "]"
end: "End" "[" "]"
countinregion: "CountInRegion" "[" INT "," INT "," INT "," INT "]"
detect: "Detect" "[" STRING ("," STRING)* "]" | "Detect" "[" "]"
segment: "Segment" "[" STRING "]"
else: "Else"
in: "IN" "[" STRING "]" EOL (INDENT expr+)*
if: "IF" "[" (expr+)* "]" EOL (INDENT expr+)* (EOL | EOF | " ") (else EOL (INDENT expr+)* (EOL | EOF | " "))?
tutorial: "Tutorial" "[" STRING "]"
OPERAND: "+" | "-" | "*" | "/"
EQUALITY: "=="
train: "Train" "[" STRING "," STRING "]" | "Train" "[" STRING "]"
label: "Label" "[" STRING "," STRING ("," STRING )*  "]" 
variable: /[a-zA-Z_][a-zA-Z0-9_]*/
comment: "#"
EOL: "\\n"
EOF: "\\Z"
INT: /-?\d+/
INDENT: "    "
BOOL: "True" | "False"
%import common.ESCAPED_STRING -> STRING
%import common.WS_INLINE
%ignore WS_INLINE
"""