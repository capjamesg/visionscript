grammar = """
start: (expr | EOL | EOF | " ")*

expr: (if | in | train | label | detect | countinregion | help | list | get | exit | read | compare | count | cutout | show | size | caption | say | save | load | use | replace | var | classify | segment | comment | contains | if | else | end | make | run | isita | find | describe | import | rotate | getcolours | getcolors | get_text | greyscale | select | paste | pasterandom | resize | blur | literal | setbrightness | search | similarity | readqr | reset | negate | BOOL | INT | equality | not_equality | input | deploy | getedges | setconfidence | setregion | filterbyclass)
classify: "Classify" "[" STRING ("," STRING)* "]"
var: variable "=" expr
replace: "Replace" "[" STRING "]"
use: "Use" "[" STRING "]"
load: "Load" "[" (STRING | input) "]" | "Load" ("[" "]")?
save: "Save" "[" STRING "]"
say: "Say" "[" STRING "]" | "Say" ("[" "]")?
get_text: "GetText" ("[" "]")?
greyscale: "Greyscale" ("[" "]")?
search: "Search" "[" STRING "]"
deploy: "Deploy" "[" STRING "]"
getedges: "GetEdges"  ("[" "]")?
filterbyclass: "FilterByClass" "[" STRING ("," STRING)* "]" | "FilterByClass" ("[" "]")?
describe: "Describe" ("[" "]")?
setregion: "SetRegion" "[" INT "," INT "," INT "," INT "]" | "SetRegion" ("[" "]")?
readqr: "ReadQR" ("[" "]")?
setconfidence: "SetConfidence" "[" FLOAT "]" | "SetConfidence" ("[" "]")?
rotate: "Rotate" "[" (INT | STRING) "]"
resize: "Resize" "[" INT "," INT "]"
getcolors: "GetColors" ("[" "]")? | "GetColors" "[" INT "]"
getcolours: "GetColours" ("[" "]")? | "GetColours" "[" INT "]"
isita: "Is it a " (("," STRING)* | ("or" STRING)*)? EOL
find: "Find" "[" STRING "]"
args: ((STRING | INT | FLOAT | expr) ("," (STRING | INT | FLOAT | expr))*) | (STRING | INT | FLOAT | expr)?
make: "Make" literal ("[" args "]")? EOL (INDENT expr+)* EOL
caption: "Caption" ("[" "]")?
size: "Size" ("[" "]")?
import: "Import" "[" STRING "]"
run: "Run" "[" STRING "]"
show: "Show" ("[" "]")?
select: "Select" ("[" "]")? | "Select" "[" INT "]"
paste: "Paste" "[" INT "," INT "]"
pasterandom: "PasteRandom" ("[" "]")?
cutout: "Cutout" ("[" "]")?
count: "Count" ("[" "]")?
input: "Input" ("[" STRING "]")?
contains: "Contains" "[" STRING "]"
compare: "Compare" ("[" "]")?
setbrightness: "SetBrightness" "[" INT "]"
read: "Read" ("[" "]")?
exit: "Exit" ("[" "]")?
blur: "Blur" ("[" "]")?
similarity: "Similarity" ("[" (INT | FLOAT) "]")?
get: "Get" "[" INT "]" EOL
list: "[" ((STRING | INT | expr) "," | (STRING | INT | expr) )* "]" EOL
help: "Help" "[" STRING "]"
end: "End" ("[" "]")?
countinregion: "CountInRegion" "[" INT "," INT "," INT "," INT "]"
detect: "Detect" "[" input "]" | "Detect" "[" STRING ("," STRING)* "]" | "Detect" ("[" "]")?
segment: "Segment" "[" STRING "]"
else: "Else"
in: "In" "[" STRING "]" EOL (INDENT expr+)* EOL 
if: "If" "[" (expr+)* "]" EOL (INDENT expr+)* (EOL | EOF) (else EOL (INDENT expr+)* (EOL | EOF | " "))?
reset: "Reset" ("[" "]")?
negate: "Not" "[" expr "]"
OPERAND: "+" | "-" | "*" | "/"
equality: (INT | STRING | expr | FLOAT) "==" (INT | STRING | expr | FLOAT)
not_equality: (INT | STRING | expr) "!=" (INT | STRING | expr)
train: "Train" "[" STRING "," STRING "]" | "Train" "[" STRING "]"
label: "Label" "[" STRING "," STRING ("," STRING )*  "]"
literal: /([a-z][a-zA-Z0-9_]*)/ ( "[" (STRING | INT | FLOAT | expr) ("," (STRING | INT | FLOAT | expr))* "]" )? | /([a-z][a-zA-Z0-9_]*)/ "[" "]"
variable: /[a-zA-Z_][a-zA-Z0-9_]*/
comment: /#.*?\\n/
EOL: "\\n"
EOF: "\\Z"
INT: /-?\d+/
FLOAT: /-?\d+\.\d+/
INDENT: "    " | "\\t"
BOOL: "True" | "False"
%import common.ESCAPED_STRING -> STRING
%import common.WS_INLINE
%ignore WS_INLINE
"""
