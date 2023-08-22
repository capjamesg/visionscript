grammar = """
start: (expr | EOL)*

expr: (set | var | make | in | if | break | random | train | label | detect | countinregion | help | get | exit | read | compare | count | cutout | show | size | caption | say | save | load | use | replace | var | classify | segment | comment | contains | if | else | end | run | isita | find | describe | import | rotate | getcolours | getcolors | get_text | greyscale | select | paste | pasterandom | resize | blur | literal | setbrightness | search | similarity | readqr | reset | negate | BOOL | INT | equality | not_equality | input | deploy | getedges | setconfidence | setregion | filterbyclass | crop | shuffle | grid | run | camera | showtext | getfps | gt | lt | expr | increment | decrement | track | getdistinctscenes | getuniqueappearances | usecamera | breakpoint | profile | math | first | last | is | merge | remove | web | wait | opposite | detectpose | comparepose | associative_array | list | STRING | EOL)
classify: "Classify" "[" STRING ("," STRING)* "]"
merge: "Merge" "[" (variable | list | associative_array) ("," (variable | list | associative_array))* "]"
var: variable "=" (expr | STRING | INT)
replace: "Replace" "[" STRING "]"
use: "Use" "[" STRING "]"
load: "Load" "[" (STRING | input) "]" | "Load[]"
save: "Save" "[" STRING "]"
getfps: "GetFPS[]"
is: "Is" "[" (expr | STRING) "]" | "Is[]"
getdistinctscenes: "GetDistinctScenes[]"
getuniqueappearances: "GetUniqueAppearances" ("[" STRING "]")?
say: "Say" "[" (STRING | math | variable | expr) "]" | "Say[]"
get_text: "GetText[]"
camera: "Camera[]"
greyscale: "Greyscale" "[]"
opposite: "Opposite" "[" (BOOL | expr) "]"
detectpose: "DetectPose[]"
comparepose: "ComparePose" "[" (expr) "," (expr) "]" | "ComparePose[]"
showtext: "ShowText" "[" (STRING | expr) "]" | "ShowText[]"
random: "Random" "[" (STRING | expr) ("," (STRING | expr))* "]"
search: "Search" "[" STRING "]"
deploy: "Deploy" "[" STRING "]"
getedges: "GetEdges"  ("[" "]")?
profile: "Profile[]"
filterbyclass: "FilterByClass" "[" STRING ("," STRING)* "]" | "FilterByClass[]"
describe: "Describe[]"
setregion: "SetRegion" "[" INT "," INT "," INT "," INT "]" | "SetRegion[]"
readqr: "ReadQR[]"
setconfidence: "SetConfidence" "[" FLOAT "]" | "SetConfidence[]"
rotate: "Rotate" "[" (INT | STRING) "]"
resize: "Resize" "[" INT "," INT "]"
getcolors: "GetColors[]" | "GetColors" "[" INT "]"
getcolours: "GetColours[]" | "GetColours" "[" INT "]"
isita: "Is it a " (("," STRING)* | ("or" STRING)*)? EOL
find: "Find" "[" STRING "]"
args: ((STRING | INT | FLOAT | expr) ("," (STRING | INT | FLOAT | expr))*) | (STRING | INT | FLOAT | expr)?
make: "Make " literal ("[" args "]") EOL (INDENT (expr+))* "End" EOL
caption: "Caption[]"
size: "Size[]"
import: "Import" "[" STRING "]"
run: "Run" "[" "]"
shuffle: "Shuffle[]"
grid: "Grid" ("[" INT "]")?
show: "Show[]"
select: "Select[]" | "Select" "[" INT "]"
paste: "Paste" "[" INT "," INT "]"
pasterandom: "PasteRandom[]"
cutout: "Cutout[]"
crop: "Crop" ("[" (INT | STRING) "," (INT | STRING) "," (INT | STRING) "," (INT | STRING) "]")?
count: "Count" ("[" STRING "]")? | "Count[]"
input: "Input" ("[" STRING "]")?
contains: "Contains" "[" STRING "]"
compare: "Compare[]"
setbrightness: "SetBrightness" "[" INT "]"
read: "Read" "[" STRING "]" | "Read[]"
exit: "Exit[]"
blur: "Blur[]"
similarity: "Similarity" ("[" (INT | FLOAT) "]")?
get: "Get" "[" (INT | expr) ("," (STRING))* "]"
set: "Set" "[" (INT | expr) ("," (STRING))* "]"
remove: "Remove" "[" (variable) ("," (expr))* "]"
help: "Help" "[" STRING "]"
end: "End[]"
wait: "Wait" "[" (INT | expr) "]"
track: "Track[]"
countinregion: "CountInRegion" "[" INT "," INT "," INT "," INT "]" | "CountInRegion" "[" STRING "]"
detect: "Detect" "[" STRING "]" | "Detect" "[" expr "]" | "Detect[]"
segment: "Segment" "[" STRING "]"
else: "Else"
breakpoint: "Breakpoint[]"
usecamera: "UseCamera[]" EOL expr+ "EndCamera" EOL
in: "In" "[" (STRING | expr) "]" EOL expr+ "Endin" EOL
if: "If" "[" comparison_expressions "]" (EOL)? expr+ "End" EOL
web: "Web" "[" (STRING | expr) ("," (STRING | expr))? "]"
reset: "Reset[]"
negate: "Not" "[" expr "]"
math: expr (OPERAND expr)*
OPERAND: "+" | "-" | "*" | "/" | "^"
equality: (expr | INT | STRING | FLOAT | BOOL) "==" (expr INT | STRING | FLOAT | BOOL)
not_equality: (INT | STRING | expr | BOOL) "!=" (INT | STRING | expr | BOOL)
train: "Train" "[" STRING "," STRING "]" | "Train" "[" STRING "]"
label: "Label" "[" STRING "," STRING ("," STRING )*  "]"
break: "Break[]"
associative_expr: STRING ":" (STRING | INT | FLOAT | expr)
associative_array: "[" EOL? (associative_expr ("," EOL? associative_expr)*)? EOL? "]" | "AssociativeArray[]"
list: "[" EOL? (INT | STRING | FLOAT | expr) ("," EOL? (STRING | INT | FLOAT | expr))*  EOL?"]" | "List[]"
first: "First[]" | "First" "[" (variable | expr) "]"
last: "Last[]" | "Last" "[" (variable | expr) "]"
literal: /([a-z][a-zA-Z0-9_]*)/ ( "[" (STRING | INT | FLOAT | expr) ("," (STRING | INT | FLOAT | expr))* "]" )? | /([a-z][a-zA-Z0-9_]*)/ "[" "]"
variable: /[a-zA-Z_][a-zA-Z0-9_]*/
comment: /#.*?\\n/
comparison_expressions: gt | lt | gte | lte | equality | not_equality
gt: expr ">" expr
lt: expr "<" expr
gte: expr ">=" expr
lte: expr "<=" expr
increment: variable "++"
decrement: variable "--"
EOL: "\\n"
EOF: "\\Z"
INT: /-?\d+/
FLOAT: /-?\d+\.\d+/
INDENT: "    " | "\\t"
BOOL: "True" | "False"
%import common.ESCAPED_STRING -> STRING
%import common.WS_INLINE
%ignore WS_INLINE
%ignore EOL
"""
