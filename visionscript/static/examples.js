
var examples = {
    "Classify an image": `Load["./folder/abbey.jpg"]
Classify["person", "cookie"]
Say[]`,
    "Detect objects in an image": `Load["./folder/abbey.jpg"]
Detect["person"]
Say[]`,
    "Replace objects with a blue box": `Load["./folder/abbey.jpg"]
Detect["car"]
Replace["blue"]
Show[]`,
    "Build a search engine": `In["./folder"]
\tLoad[]
Search["taylor swift"]
Compare[]`,

    "Count hot dogs in an image": `Load["./folder/abbey.jpg"]
Detect["hot dogs"]
Count[]
Say[]`,
    "Rotate and greyscale an image": `Load["./folder/abbey.jpg"]
Rotate[90]
Greyscale[]
Say[]`
};

var examples_select = document.getElementById("examples");

examples_select.addEventListener("change", function (event) {
    var example = examples_select.value;
    var code = examples[example];
    textarea.value = code;
});