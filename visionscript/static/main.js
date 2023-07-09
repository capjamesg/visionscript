var mode = "interactive";

function switch_mode () {
    if (mode == "interactive") {
        document.getElementById("mode_switch").innerText = "Use Interactive Mode 📝";
        mode = "code";
    } else {
        document.getElementById("mode_switch").innerText = "Use Code Mode 📝";
        mode = "interactive";
    }
    var interactive_mode = document.getElementById("interactive_mode");
    var code_mode = document.getElementById("code_mode");
    if (mode == "interactive") {
        interactive_mode.style.display = "block";
        code_mode.style.display = "none";
    } else {
        interactive_mode.style.display = "none";
        code_mode.style.display = "block";
    }
}

var mapped_functions = {};
        
for (var category in FUNCTIONS) {
    var functions = FUNCTIONS[category];
    for (var name in functions) {
        mapped_functions[name] = functions[name];
    }
}

var search = document.getElementById("search");

// search functions
function processSearch() {
    var search = document.getElementById("search");
    var query = search.value.toLowerCase();
    var functions = document.getElementsByClassName("function");
    for (var i = 0; i < functions.length; i++) {
        var function_element = functions[i];
        var function_name = function_element.id.toLowerCase();
        if (function_name.includes(query)) {
            function_element.style.display = "block";
        } else {
            function_element.style.display = "none";
        }
    }
}

// if user hits enter, take top result
// listen for key
document.addEventListener("keydown", function (event) {
    var search = document.getElementById("search");

    if (search != document.activeElement) {
        return;
    }

    console.log(event.key);

    if (event.key == "Enter") {
        // set background color
        document.getElementById("drag_drop_notebook").style.background = "white";
        var functions = document.getElementsByClassName("function");
        var function_element = functions[0];
        var function_name = function_element.id;
        var code = function_name + "[]";
        var color = function_element.firstElementChild.style.color;

        var html = "";

        var cell_count = cells.children.length + 1;

        html = `
            <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color};">
                <p>${function_name}[]</p>
            </div>
        `;

        notebook.appendChild(document.createRange().createContextualFragment(html));
    }
});

function importNotebookToInteractiveCode (notebook) {

}

var colors = {
    "input": "#a2d2ff",
    "process": "#cdb4db",
    "find": "#ccd5ae",
    "output": "lavender",
    "logic": "#ffdf6b"
};

for (var category in FUNCTIONS) {
    var functions = FUNCTIONS[category];
    var function_box = document.getElementById("function_box");
    function_box.innerHTML += `
        <h2>${category}</h2>
    `;
    for (var name in functions) {
        var args = functions[name].args;
        var description = functions[name].description;
        var example = functions[name].example;
        function_box.innerHTML += `
            <div class="function" draggable="true" id="${name}">
                <p style="color: ${colors[category]}">${name}[${args.join(", ")}]</p>
                <p>${description}</p>
            </div>
        `;
        functions[name].element = document.getElementById(name);
    }
}

// ecah function is draggable
var functions = document.getElementsByClassName("function");

for (var i = 0; i < functions.length; i++) {
    var function_element = functions[i];
    function_element.addEventListener("dragstart", function (event) {
        event.dataTransfer.setData("text/plain", event.target.id);
    });
}

var notebook = document.getElementById("drag_drop_notebook");

notebook.addEventListener("dragover", function (event) {
    event.preventDefault();
});

notebook.addEventListener("dragleave", function (event) {
    event.preventDefault();
    notebook.style.backgroundColor = "white";
});

notebook.addEventListener("drop", function (event) {
    // remove drag_drop_notebook background
    notebook.style.background = "none";
    event.preventDefault();
    // if hovered over another element, add a 20px margin
    var cells = document.getElementById("drag_drop_notebook");
    // if cell has an argumetn_block, don't do anything
    // if hovered outside of drag_drop_notebook, delete
    if (cells.children.length > 0) {
        if (cells.children[cells.children.length - 1].id == "argument_block") {
            return;
        }
    }
    // get "text/plain" cata
    var function_name = event.dataTransfer.getData("text/plain");
    var function_element = document.getElementById(function_name);

    // if is cell, don't do anything
    if (function_element.classList.contains("cell")) {
        console.log("is cell");
        // move cell in list
        var cell = function_element;
        var cell_index = Array.prototype.indexOf.call(cells.children, cell);
        var cell_count = cells.children.length;
        // move under closest cell
        var closest_cell = null;
        var closest_cell_index = null;
        var closest_cell_distance = null;

        for (var i = 0; i < cell_count; i++) {
            var other_cell = cells.children[i];
            if (other_cell == cell) {
                continue;
            }
            var other_cell_rect = other_cell.getBoundingClientRect();
            var notebook_rect = notebook.getBoundingClientRect();
            var distance = Math.abs(event.clientY - other_cell_rect.top);
            if (closest_cell == null) {
                closest_cell = other_cell;
                closest_cell_index = i;
                closest_cell_distance = distance;
            } else {
                if (distance < closest_cell_distance) {
                    closest_cell = other_cell;
                    closest_cell_index = i;
                    closest_cell_distance = distance;
                }
            }
        }

        if (closest_cell == null) {
            return;
        }

        if (closest_cell_index > cell_index) {
            // move cell down
            cells.insertBefore(cell, closest_cell.nextSibling);
        } else {
            // move cell up
            cells.insertBefore(cell, closest_cell);
        }
        return;
    }
    // last cell
    var margin = 20;
    var nested = false;
    // con
    if (cells.children.length > 0) {
        // if last cell function was an If or In
        if (cells.children[cells.children.length - 1].id.includes("If") || cells.children[cells.children.length - 1].id.includes("In")) {
            var last_cell = cells.children[cells.children.length - 1];
            var last_cell_rect = last_cell.getBoundingClientRect();
            var notebook_rect = notebook.getBoundingClientRect();
            console.log(event.clientY);
            if (event.clientY > last_cell_rect.top && event.clientY < last_cell_rect.bottom) {
                margin = 40;
                nested = true;
            }
        }
    }

    // if target cell is an If statement, don't add input
    console.log(event.target.id);
    if (event.target.id.includes("If")) {
        var function_name = event.dataTransfer.getData("text/plain");
        var function_element = document.getElementById(function_name);
        var code = function_name + "[]";
        var color = function_element.firstElementChild.style.color;

        var html = "";

        var cell_count = cells.children.length + 1;

        html = `
            <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: ${margin}px;">
                <p>${function_name} <div class="argument_block">[]</div></p>
            </div>
        `;

        notebook.appendChild(document.createRange().createContextualFragment(html));
        return;
    }
    
    var function_name = event.dataTransfer.getData("text/plain");
    var function_element = document.getElementById(function_name);
    var code = function_name + "[]";
    var color = function_element.firstElementChild.style.color;

    var html = "";

    var cell_count = cells.children.length + 1;

    console.log(function_name)

    if (mapped_functions[function_name].supports_arguments) {
        // if it is an if statement, don't add input
        if (function_name != "If" || function_name != "Input") {
            html = `
                <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: ${margin}px;">
                    <p>${function_name}[<input type="text" class="argument_block" id="cell_${cell_count}" />]</p>
                </div>
            `;
        } else {
            html = `
                <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: ${margin}px;">
                    <p>${function_name} <div class="argument_block">[]</div></p>
                </div>
            `;
        }
    } else {
        html = `
            <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: ${margin}px;">
                <p>${function_name}[]</p>
            </div>
        `;
    }

    notebook.appendChild(document.createRange().createContextualFragment(html));

    if (nested) {
        var cell = document.getElementById(`${function_name}_${cell_count}`);
        cell.dataset.nested = true;
    }

    var cell = document.getElementById(`${function_name}_${cell_count}`);
    cell.addEventListener("dragstart", function (event) {
        event.dataTransfer.setData("text/plain", event.target.id);
    });

    // if supports arguments, allow drag and drop in box to nest
    if (mapped_functions[function_name].supports_arguments) {
        var argument_block = document.getElementsByClassName("argument_block");
        var argument_block = argument_block[argument_block.length - 1];
        argument_block.addEventListener("dragover", function (event) {
            event.preventDefault();
        });
        argument_block.addEventListener("dragleave", function (event) {
            event.preventDefault();
            argument_block.style.backgroundColor = "white";
        });
        argument_block.addEventListener("drop", function (event) {
            // replace <input> with Input[]
            var function_name = event.dataTransfer.getData("text/plain");
            var function_element = document.getElementById(function_name);
            var code = function_name + "[]";
            // if it is a cell, don't do anything
            // replace event.target with Input[]
            if (function_name == "Input") {
                var argument_block = event.target;
                // replace argument_block
                var argument_block = document.getElementById("cell_" + argument_block.id.split("_")[1]);
                var input_field = event.target;
                console.log(input_field);
                // replace with p
                var p = document.createElement("p");
                p.innerText = "Input[]";
                p.style.display = "inline-block";
                p.style.margin = "0";
                p.style.padding = "0";
                p.style.backgroundColor = "white";
                p.classList.add("argument_block");

                // create input and append to argument block
                var input = document.createElement("input");
                input.type = "text";
                input.classList.add("argument_block");
                input.id = "cell_" + argument_block.id.split("_")[1];
                input.style.display = "inline-block";
                input.style.margin = "0";
                input.style.padding = "0";
                input.style.backgroundColor = "white";
                p.appendChild(input);
                // replace
                argument_block.replaceWith(p);
            }
        });
    }
});

function getCodeFromInteractiveEnvironment () {
    var code = "";
    var functions = document.getElementsByClassName("cell");

    var in_nested_context = false;
    for (var i = 0; i < functions.length; i++) {
        var function_element = functions[i];
        var function_name = function_element.id.split("_")[0];

        // this is the form to create new elements
        if (function_name == "new") {
            continue;
        }

        var argument_block = function_element.getElementsByClassName("argument_block");

        console.log(argument_block, "iiiii");

        if (argument_block.length > 0) {
            var argument_block = argument_block[0];
            var argument = argument_block.value;
            // surround args in quotes

            // if data-filename, replace with file name
            if (argument) {
                argument = argument;
            } else if (argument_block.dataset.filename) {
                argument = argument_block.dataset.filename;
            } else if (argument == "") {
                argument = mapped_functions[function_name].argument_default;
            // if there is a child input, get value
            } else if (argument_block.children.length > 0) {
                var first_arg = argument_block.innerText;
                // remove last two cars
                console.log(first_arg, argument);
                first_arg = first_arg.substring(0, first_arg.length - 2);
                argument = first_arg + "[\"" + argument_block.children[0].value + "\"]";
            } else {
                argument = argument_block.innerText;
            }

            // add "" between commas
            var args = argument.split(",");

            for (var j = 0; j < args.length; j++) {
                var arg = args[j];
                if (arg == "") {
                    continue;
                }
                if (!arg.startsWith("Input[")) {
                    args[j] = "\"" + arg + "\"";
                }
            }

            argument = args.join(", ");

            argument = argument.replace(/\"\"/g, "");

            // if nested, start w/ indent
            if (function_element.dataset.nested) {
                code += "    ";
                in_nested_context = true;
            }

            code += function_name + "[" + argument + " ]" + "\n";
        } else {
            code += function_name + "[]" + "\n";
        }
    }

    // if code doesn't end with Say[], add it
    if (!code.endsWith("Say[]\n")) {
        code += "Say[]\n";
    }

    return code;
}

var run = document.getElementById("run");

run.addEventListener("click", function (event) {
    event.preventDefault();
    var output = document.getElementById("output");
    var code = getCodeFromInteractiveEnvironment();
    executeCode(code);
});

function deploy () {
    var deploy = document.getElementById("deploy");
    deploy.showModal();
    var deploy_form = document.getElementById("deploy_form");
    deploy_form.addEventListener("submit", function (event) {
        event.preventDefault();
        var data = new FormData(deploy_form);
        var name = data.get("name");
        fetch('http://localhost:5000/deploy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({name: name})
        })
        .then(response => response.json())
        .then(data => {
            var deploy_message = document.getElementById("deploy_message");
            deploy_message.innerText = data.message;
        })
        .catch((error) => {
            var deploy_message = document.getElementById("deploy_message");
            deploy_message.innerText = "Your app could not be deployed. Please make sure your app code is valid.";
        });
    });
}
var cells = document.getElementById("cells");

function startLoading(loading) {
    var timer = setInterval(function () {
        if (loading.innerText == "Loading") {
            loading.innerText = "Loading.";
        } else if (loading.innerText == "Loading.") {
            loading.innerText = "Loading..";
        } else if (loading.innerText == "Loading..") {
            loading.innerText = "Loading...";
        } else if (loading.innerText == "Loading...") {
            loading.innerText = "Loading";
        } else {
            loading.innerText = "Loading";
        }
    }, 500);
    return timer;
}

function executeCode (code) {
    // make loading wheel

    cells.innerHTML += `
        <li class="cell" id="loading">Loading</li>
    `;
    var loading = document.getElementById("loading");
    var output = document.getElementById("output");
    // show output
    output.style.display = "block";
    var timer = startLoading(loading);
    var output_timer = startLoading(output);

    var error_cell = document.getElementById("error");
    
    error_cell.innerText = "";
    error_cell.style.display = "none";

    fetch('http://localhost:5000/notebook', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({code:
            code,
            state_id: "{{ state_id }}"
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        if (data.output == null) {
            data.output = "";
        }
        if (data.output.image) {
            data.output = `<img src="data:image/png;base64,${data.output.image}">`;
        }
        var time = data.time;
        // delete loading cell
        cells.removeChild(loading);
        clearInterval(timer);
        clearInterval(output_timer);

        var row_count = (code.match(/\n/g) || []).length + 1;

        // if interactive mode, show in #output
        if (mode == "interactive") {
            var output = document.getElementById("output");
            output.innerHTML = data.output;
            return;
        }

        cells.innerHTML += `
            <li class="cell">
                <p class="time">#${cells.children.length + 1} (${time}s)</p>
                <textarea rows="${row_count}" readonly class="cell_run">${code}</textarea>
                <pre ${data.error ? 'class="error_cell"' : ''}>${data.error ? data.error : data.output}</pre>
            </li>
        `;
        
        console.log(cells.children.length);
        document.getElementById("current_count").innerHTML = `#${cells.children.length + 1}`;

        // click to copy to clipboard
        for (var i = 0; i < cells.children.length; i++) {
            var cell = cells.children[i];
            var textarea = cell.getElementsByTagName("textarea")[0];
            textarea.addEventListener("click", function (event) {
                event.preventDefault();
                var text = event.target.value;
                navigator.clipboard.writeText(text);
                alert("Copied to clipboard");
            });
        }
    })
    .catch((error) => {
        clearInterval(timer);
        clearInterval(output_timer);
        
        var error_cell = document.getElementById("error");
        error_cell.innerText = "There was an error running your code. Please make sure your code is valid.";
        error_cell.style.display = "block";
        // hide output
        output.style.display = "none";
    });
}

var form = document.getElementById("new");

form.addEventListener("submit", function (event) {
    event.preventDefault();
    var data = new FormData(form);
    var code = data.get("jscode");
    executeCode(code);
});

// auto-expand textarea
var textarea = document.getElementById("jscode");

textarea.addEventListener("input", function (event) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
});

var export_vic = document.getElementById("export_vic");

export_vic.addEventListener("click", function (event) {
    event.preventDefault();
    var data = new FormData(form);
    if (mode == "interactive") {
        var code = getCodeFromInteractiveEnvironment();
        data.set("jscode", code);
    }
    var code = data.get("jscode");
    var blob = new Blob([code], {type: "text/plain;charset=utf-8"});
    // download
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "notebook.vic";
    a.click();
});

var dropzone = document.getElementsByTagName("body")[0];

dropzone.addEventListener("dragover", function (event) {
    event.preventDefault();
});

dropzone.addEventListener("dragleave", function (event) {
    event.preventDefault();
    dropzone.style.backgroundColor = "white";
});

// function process_image_drop (event)

dropzone.addEventListener("drop", function (event) {
    if (event.target.id == "drag_drop_notebook") {
        return;
    }
    // if function box, delete the function box
    // read data channel
    var readData = event.dataTransfer.getData("text/plain");
    if (readData) {
        // delete element
        var element = document.getElementById(readData);
        element.parentNode.removeChild(element);
        return;
    }
    event.preventDefault();
    dropzone.style.backgroundColor = "white";
    var file = event.dataTransfer.files[0];
    var body = new FormData();
    body.append("file", file)
    body.append("state_id", "{{ state_id }}");
    // base64 file
    var reader = new FileReader();
    // read file
    reader.readAsDataURL(file);

    // only allow jpeg, jpg, png, or .vicnb
    console.log(file.name);
    if (!file.name.endsWith(".jpg") && !file.name.endsWith(".jpeg") && !file.name.endsWith(".png") && !file.name.endsWith(".vicnb")) {
        var dialog = document.getElementById("dialog");
        console.log(dialog, "e");
        var error_message = document.getElementById("error_message");
        error_message.innerText = "Your file could not be uploaded. Please make sure you have uploaded a supported format.";
        dialog.showModal();
        return;
    }

    // post to /notebook/upload with state id
    fetch('http://localhost:5000/notebook/upload?state_id={{ state_id }}', {
        method: 'POST',
        body: body
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        if (data.cells) {
            if (mode == "code") {
                data.cells.forEach(function (cell) {
                    var code = cell.cell;
                    var output = cell.output;
                    if (output == null) {
                        output = "";
                    }
                    if (output.image) {
                        output = `<img src="data:image/png;base64,${output.image}">`;
                    }
                    cells.innerHTML += `
                        <li class="cell">
                            <p>#${cells.children.length + 1}</p>
                            <textarea rows="3" disabled>${code}</textarea>
                            <pre>${output}</pre>
                        </li>
                    `;
                });
                return;
            } else {
                var interactive_notebook = document.getElementById("drag_drop_notebook");

                data.cells.forEach(function (cell) {
                    var code = cell.cell;
                    // for item in newline
                    var code = code.split("\n");
                    for (var i = 0; i < code.length; i++) {
                        // set background to white
                        interactive_notebook.style.background = "white";
                        var line = code[i];
                        var function_name = line.split("[")[0];
                        var argument = line.split("[")[1];
                        if (argument) {
                            argument = argument.split("]")[0];
                        }
                        console.log(function_name);
                        var color = mapped_functions[function_name].element.firstElementChild.style.color;
                        var html = "";
                        if (mapped_functions[function_name].supports_arguments) {
                            html = `
                                <div class="cell" draggable="true" id="${function_name}_${i + 1}" style="background-color: ${color}; margin-left: 20px;">
                                    <p>${function_name}[<input type="text" class="argument_block" id="cell_${i + 1}" value="${argument}">]</p>
                                </div>
                            `;
                        } else {
                            html = `
                                <div class="cell" draggable="true" id="${function_name}_${i + 1}" style="background-color: ${color}; margin-left: 20px;">
                                    <p>${function_name}[]</p>
                                </div>
                            `;
                        }
                        interactive_notebook.appendChild(document.createRange().createContextualFragment(html));
                    }
                }).catch((error) => {
                    return;
                });
            }
        }
        var file_name = data.file_name;
        var files = document.getElementById("files");
        var files_section = document.getElementById("files_section");

        files_section.style.display = "block";

        var base64 = reader.result;

        // if already exists, don't add
        var file_names = document.getElementsByClassName("file_name");

        for (var i = 0; i < file_names.length; i++) {
            var file_name_element = file_names[i];
            console.log(file_name_element.innerText, file_name);
            if (file_name_element.innerText == file_name) {
                return;
            }
        }

        files.innerHTML += `
            <li><img src="${base64}" alt="${file_name}" height=100 width=100 data-filename="${file_name}" style="display: block;">${file_name}</li>
        `;
        // if dragged over an Load statement, add iamge to the argument block
        
        if (event.target.classList.contains("argument_block")) {
            // replace argument block
            var argument_block = event.target;
            var new_element = `
                <img src="${base64}" alt="${file_name}" height=100 width=100 class="argument_block" data-filename="${file_name}">
            `;

            // add before argument block
            argument_block.insertAdjacentHTML("beforebegin", new_element);

            // remove argument block
            argument_block.parentNode.removeChild(argument_block);
        }
    })
    .catch(err => {
        var dialog = document.getElementById("dialog");
        var error_message = document.getElementById("error_message");
        // if file ends with .vicnb
        console.log(err);
        if (file.name.endsWith(".vicnb")) {
            error_message.innerText = "Please import your notebook in interactive mode.";
            dialog.showModal();
            return;
        }

        error_message.innerText = "Your file could not be uploaded. Please make sure you have uploaded a supported format.";
        dialog.showModal();
    });
});

function resetNotebook() {
    var code = "Reset[]\n";
    executeCode(code);
    // show dialog
    var dialog = document.getElementById("dialog");
    var error_message = document.getElementById("error_message");
    error_message.innerText = "Your notebook has been reset.";
    dialog.showModal();
}

var export_vicnb = document.getElementById("export_vicnb");

export_vicnb.addEventListener("click", function (event) {
    event.preventDefault();
    fetch('http://localhost:5000/notebook/save?state_id={{ state_id }}', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        var blob = new Blob([JSON.stringify(data.file)], {type: "text/plain;charset=utf-8"});
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "notebook.vicnb";
        a.click();
    })
    .catch(err => {
        console.log(err);
    });
});