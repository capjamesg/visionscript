var mode = "interactive";
var highlighted_category = "Input";

function hideFunctionBox () {
    var function_box = document.getElementById("function_box");
    var toggle = document.getElementById("hide_function_box");
    // hide ul in function box
    var ul = function_box.getElementsByTagName("div");
    for (var i = 0; i < ul.length; i++) {
        if (!ul[i].style.display || ul[i].style.display == "block") {
            ul[i].style.display = "none";
            // set function box height to 100 px
            function_box.style.height = "100px";
            // hide search
            var search = document.getElementById("search");
            search.style.display = "none";
            toggle.innerText = "Show Blocks";
        } else {
            ul[i].style.display = "block";
            // set function box height to 300 px
            function_box.style.height = "300px";
            // show search
            var search = document.getElementById("search");
            search.style.display = "block";
            toggle.innerText = "Hide Blocks";
        }
    }
}

function switch_paper_mode () {
    var dialog = document.getElementById("paper_mode_dialog");

    if (dialog.style.display == "none") {
        // disable webcam
        var paper_mode_video = document.getElementById("paper_mode_video");
        paper_mode_video.pause();
        paper_mode_video.srcObject = null;
        
        dialog.close();
    } else {
        dialog.showModal();
        start_paper_mode_camera();
    }   
}

function start_paper_mode_camera () {
    var paper_mode_video = document.getElementById("paper_mode_video");

    // use front-facing camera
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } }).then(function (stream) {
        paper_mode_video.srcObject = stream;
        paper_mode_video.play();
        paper_mode_video.style.display = "block";
    });
}

function take_photo () {
    var paper_mode_video = document.getElementById("paper_mode_video");
    var canvas = document.getElementById("paper_mode_canvas");
    canvas.width = paper_mode_video.videoWidth;
    canvas.height = paper_mode_video.videoHeight;

    var context = canvas.getContext("2d");
    // take full resolution photo
    context.drawImage(paper_mode_video, 0, 0, canvas.width, canvas.height);
    // hide video
    paper_mode_video.style.display = "none";
    // show canvas
    canvas.style.display = "block";

    // paint image
    var image = document.getElementById("paper_mode_image");
    image.src = canvas.toDataURL("image/png");

    var paper_mode_video = document.getElementById("paper_mode_video");
    paper_mode_video.style.display = "none";

    // show retake_paper_mode_button
    var retake_paper_mode_button = document.getElementById("retake_paper_mode_button");
    retake_paper_mode_button.style.display = "block";

    // hide paper_mode_button
    var paper_mode_button = document.getElementById("paper_mode_button");
    paper_mode_button.style.display = "none";

    // show run_paper_mode
    var run_paper_mode = document.getElementById("run_paper_mode");
    run_paper_mode.style.display = "block";
}

// https://stackoverflow.com/questions/16968945/convert-base64-png-data-to-javascript-file-objects
function dataURLtoFile(dataurl, filename) {
    var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
}

function run_paper_code () {
    var paper_mode_image = document.getElementById("paper_mode_image");

    var data = paper_mode_image.src;

    var body = new FormData();
    body.append('file', dataURLtoFile(data, "paper_mode.png"));

    // set drag_drop_notebook background to image
    var drag_drop_notebook = document.getElementById("drag_drop_notebook");
    drag_drop_notebook.style.backgroundImage = `url(${data})`;

    // upload image
    fetch(`${API_URL}/notebook/upload?state_id=${STATE_ID}`, {
        method: 'POST',
        body: body
    })
    .then(response => response.json())
    .then(data => {
        var file_name = data["file_name"];

        // close modal
        var dialog = document.getElementById("paper_mode_dialog");
        dialog.close();

        // // add session id to file name
        // file_name = `tmp/${STATE_ID}/${file_name}`;

        var paper_code_program = `
        Load["${file_name}"]
        GetText[]
        Say[]
        `;
        
        executeCode(paper_code_program, false, "paper_mode");
    });
}

function rerun (cell_number) {
    var cells = document.getElementById("cells");
    var cell = cells.children[cell_number - 1];
    
    var textarea = cell.getElementsByTagName("textarea")[0];
    textarea.value = textarea.value.trim();
    var output = cell.getElementsByTagName("pre")[0];
    output.innerText = "";
    var output = document.getElementById("output");
    output.style.display = "block";

    executeCode(textarea.value, false, cell.id);
}

function show_toast (message) {
    var toast = document.getElementById("toast");
    toast.innerText = message;
    toast.style.display = "block";
    setTimeout(function () {
        toast.style.display = "none";
    }, 3000);
}

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
        // and functoin has data-category of category
        if (function_name.includes(query) && (function_element.dataset.category == highlighted_category || highlighted_category == "All")) {
            function_element.style.display = "block";
        } else {
            function_element.style.display = "none";
        }
    }
}


document.addEventListener("keydown", function (event) {
    // Command + Enter should run
    if (event.code == "Enter" && event.metaKey) {
        var code = getCodeFromInteractiveEnvironment();
        executeCode(code);
        return;
    }
    // Command + S should save
    if (event.key == "s" && event.metaKey) {
        event.preventDefault();
        export_vicnb();
        return;
    }
});

// if user hits enter, take top result
// listen for key
document.addEventListener("keydown", function (event) {
    var search = document.getElementById("search");

    if (search != document.activeElement) {
        return;
    }


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

        if (mapped_functions[function_name].supports_arguments) {
            html = `
                <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: 20px;">
                    <p>${function_name}[<input type="text" class="argument_block" id="cell_${cell_count}" />]</p>
                </div>
            `;
        } else {
            html = `
                <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: 20px;">
                    <p>${function_name}[]</p>
                </div>
            `;
        }

        notebook.appendChild(document.createRange().createContextualFragment(html));
    }
});

var colors = {
    "Input": "#f44336",
    "Process": "#2196f3",
    "Find": "#4caf50",
    "Output": "#ff9800",
    "Logic": "#ffdf6b",
    "Deploy": "#3f51b5"
};

for (var category in FUNCTIONS) {
    var functions = FUNCTIONS[category];
    var function_box = document.getElementById("function_box");
    for (var name in functions) {
        var args = functions[name].args;
        var description = functions[name].description;
        var example = functions[name].example;
        function_box.innerHTML += `
            <div class="function sidebar_cell" draggable="true" id="${name}" data-color="${colors[category]}" data-category="${category}" style="background-color: ${colors[category]}; opacity: 0.8;">
                <p><b>${name}[${args.join(", ")}]</b>
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

    // on mobile tap, add to notebook
    function_element.addEventListener("touchstart", function (event) {
        event.preventDefault();
        // set background color
        // ACCESS functinos from parent

        document.getElementById("drag_drop_notebook").style.background = "white";
  
        // get function name
        var function_name = event.target.id;

        // if no id, traverse up
        if (!function_name) {
            while (!function_name) {
                function_name = event.target.parentElement.id;
                // break if element is <html>
                if (event.target.parentElement.tagName == "HTML") {
                    break;
                }
            }
        }


        var function_element = document.getElementById(function_name);
        
        var category = function_element.dataset.category;
        var color = colors[category];

        var html = "";

        var cell_count = cells.children.length + 1;

        if (mapped_functions[function_name].supports_arguments) {
            html = `
                <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: 20px;">
                    <p>${function_name}[<input type="text" class="argument_block" id="cell_${cell_count}" />]</p>
                </div>
            `;
        } else {
            html = `
                <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: 20px;">
                    <p>${function_name}[]</p>
                </div>
            `;
        }

        notebook.appendChild(document.createRange().createContextualFragment(html));

        // function doubleTap (element) {
        //     var lastTap = 0;
        //     return function (event) {
        //         var currentTime = new Date().getTime();
        //         var tapLength = currentTime - lastTap;
        //         event.preventDefault();
        //         if (tapLength < 500 && tapLength > 0) {
        //             // double tap
        //             element.remove();
        //         }
        //         lastTap = currentTime;
        //     };
        // }
        // // if double tap, delete block
        // var cell = document.getElementById(`${function_name}_${cell_count}`);

        // cell.addEventListener("touchstart", doubleTap(cell));

        // if argument block, allow tap on block to upload file
        if (mapped_functions[function_name].supports_arguments && mapped_functions[function_name].args.includes("file")) {
            var argument_block = document.getElementsByClassName("argument_block");
            var argument_block = argument_block[argument_block.length - 1];
            argument_block.addEventListener("click", function (event) {
                event.preventDefault();
                var file_input = document.createElement("input");
                file_input.type = "file";
                file_input.click();
                file_input.addEventListener("change", function (event) {
                    for (var file of event.target.files) {
                        console.log(file);
                        var body = new FormData();
                        body.append("file", file)
                        body.append("state_id", STATE_ID);
                        // base64 file
                        var reader = new FileReader();
                        // read file
                        reader.readAsDataURL(file);

                        // only allow jpeg, jpg, png, or .vicnb, avif, webp
                        file.name = file.name.toLowerCase();
                        if (!file.name.endsWith(".jpg") && !file.name.endsWith(".jpeg") && !file.name.endsWith(".png") && !file.name.endsWith(".vicnb") && !file.name.endsWith(".avif") && !file.name.endsWith(".webp")) {
                            var dialog = document.getElementById("dialog");
                            var error_message = document.getElementById("error_message");
                            error_message.innerText = "Your file could not be uploaded. Please make sure you have uploaded a supported format.";
                            dialog.showModal();
                            return;
                        }

                        // post to /notebook/upload with state id
                        fetch(`${API_URL}/notebook/upload?state_id=${STATE_ID}`, {
                            method: 'POST',
                            body: body
                        })
                        .then(response => response.json())
                        .then(data => {
                            var cell = document.getElementById("cell_" + argument_block.id.split("_")[1]);
                            cell.value = data.file_name;
                            cell.dataset.filename = data.file_name;
                            var img = document.createElement("img");
                            img.src = reader.result;
                            img.style.width = "100px";
                            img.style.height = "100px";
                            argument_block.innerHTML = "";
                            img.dataset.filename = data.file_name;
                            img.classList.add("argument_block");
                            cell.replaceWith(img);

                            if (file.name.endsWith(".vicnb")) {
                                show_toast("Your notebook has been imported.");
                            } else {
                                show_toast("Your file has been uploaded.");
                            }
                        });
                    }
                });
            });
        }
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

    // get "text/plain" data
    var function_name = event.dataTransfer.getData("text/plain");
    var function_element = document.getElementById(function_name);

    // skip Input[]
    if (function_name == "Input") {
        return;
    }
    // if is cell, don't do anything
    if (function_element && function_element.classList.contains("cell")) {
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
    if (cells.children.length > 0) {
        // if last cell function was an If or In
        if (cells.children[cells.children.length - 1].id.includes("If") || cells.children[cells.children.length - 1].id.includes("In")) {
            var last_cell = cells.children[cells.children.length - 1];
            var last_cell_rect = last_cell.getBoundingClientRect();
            if (event.clientY > last_cell_rect.top && event.clientY < last_cell_rect.bottom) {
                margin = 40;
                nested = true;
            }
        }
    }

    // if target cell is an If statement, don't add input
    if (event.target.id.includes("If")) {
        var function_name = event.dataTransfer.getData("text/plain");
        var function_element = document.getElementById(function_name);
        var code = function_name + "[]";
        var category = function_element.dataset.category;
        var color = colors[category];

        var html = "";
        console.log(mapped_functions[function_name], function_name);

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
    
    var function_element = document.getElementById(function_name) || document.getElementById("Input");
    var color = function_element.firstElementChild.style.color;

    var html = "";

    var cell_count = cells.children.length + 1;
    console.log(mapped_functions[function_name], function_name);
    var category = function_element.dataset.category;
    var color = colors[category];

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
            // if it is a cell, don't do anything
            // replace event.target with Input[]
            if (function_name == "Input") {
                var argument_block = event.target;
                // replace argument_block
                var argument_block = document.getElementById("cell_" + argument_block.id.split("_")[1]);
                var input_field = event.target;
                // replace with p
                var p = document.createElement("p");
                p.innerText = "Input[";
                p.style.display = "inline-block";
                p.style.margin = "10px";
                p.style.padding = "5px;";
                p.style.backgroundColor = "white";
                p.classList.add("argument_block");

                // create input and append to argument block
                var input = document.createElement("input");
                input.type = "text";
                input.classList.add("argument_block");
                input.id = "cell_" + argument_block.id.split("_")[1];
                input.style.backgroundColor = "white";
                input.style.margin = "10px";
                // add "]" after input
                var p2 = document.createElement("p");
                p2.innerText = "]";
                p2.style.display = "inline-block";
                p2.style.margin = "0";
                p2.style.padding = "0";
                p2.style.backgroundColor = "white";
                p.appendChild(input);
                p.appendChild(p2);
                // replace
                argument_block.replaceWith(p);
                argument_block.style.margin = "10px";
            }
        });
    }
});

function getCodeFromInteractiveEnvironment () {
    var code = "";
    var functions = document.getElementsByClassName("cell");

    for (var i = 0; i < functions.length; i++) {
        var function_element = functions[i];
        var function_name = function_element.id.split("_")[0];

        // this is the form to create new elements
        if (function_name == "new") {
            continue;
        }

        var argument_block = function_element.getElementsByClassName("argument_block");

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
    if (!code.endsWith("Say[]\n") && !code.endsWith("Show[]\n")) {
        code += "Say[]\n";
    }

    return code;
}

var run = document.getElementById("run");

run.addEventListener("click", function (event) {
    event.preventDefault();
    var code = getCodeFromInteractiveEnvironment();
    // if code doesn't end with Say[], add it
    if (!code.endsWith("Say[]\n") && !code.endsWith("Show[]\n")) {
        code += "Say[]\n";
    }

    executeCode(code);
});

function deploy_code (publish_as_noninteractive_webpage) {
    fetch(`${API_URL}/notebook/deploy`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({state_id: STATE_ID, name: document.getElementById("name").value, api_url: document.getElementById("api_url").value, api_key: document.getElementById("api_key").value, description: document.getElementById("description").value, publish_as_noninteractive_webpage: publish_as_noninteractive_webpage})
    })
    .then(response => response.json())
    .then(data => {
        var deploy_message = document.getElementById("deploy_message");
        deploy_message.innerText = "Published to " + data.message;
        deploy_message.style.display = "block";
        console.log(data);
        var qrcode = data.qr_code;
        // base64
        deploy_message.innerHTML += `<p>Scan the code below to view your creation on a mobile device: <br /><br /><img src="${qrcode}" />`;
        // add success class
        deploy_message.classList.add("success");
    })
    .catch((error) => {
        var deploy_message = document.getElementById("deploy_message");
        deploy_message.innerText = "Your app could not be deployed. Please make sure your app code is valid and you have filled out the deployment form in full.";
        // show deploy_message
        deploy_message.style.display = "block";
        // add error class
        deploy_message.classList.add("error");
    });
}

function deploy () {
    var deploy = document.getElementById("deploy");
    deploy.showModal();
    var deploy_form = document.getElementById("deploy_form");
    deploy_form.addEventListener("submit", function (event) {
        event.preventDefault();
        deploy_code(false);
    });
}
var cells = document.getElementById("cells");

function startLoading(loading) {
    loading.style.display = "block";
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

function executeCode (code, comment = false, existing_cell = null) {
    // if (!code.endsWith("Say[]\n") && !code.endsWith("Show[]\n")) {
    //     code += "\n Say[]\n";
    // }

    var loading = document.getElementById("loading");
    var output = document.getElementById("output");

    output.style.display = "block";

    var timer = startLoading(loading);
    // var output_timer = startLoading(output);

    var error_cell = document.getElementById("error");
    
    error_cell.innerText = "";
    error_cell.style.display = "none";

    var is_text_cell = comment;

    var input_modal_html = ``;

    // if there are any Input[] cells in code, show modal
    if (code.includes("Input[")) {
        var fill_inputs_inputs = document.getElementById("fill_inputs_inputs");
        fill_inputs_inputs.innerHTML = "";

        var inputs = code.match(/Input\[[^\]]*\]/g);

        for (var i = 0; i < inputs.length; i++) {
            var input = inputs[i];
            // get text between ""
            var input_name = input.match(/\"[^\"]*\"/g)[0].replace(/\"/g, "");

            input_modal_html += `<div class="input"><label for="${input_name}">${input_name}</label><input type="text" id="${input_name}" name="${input_name}" /></div>`;
        }

        fill_inputs_inputs.innerHTML = input_modal_html;

        var fill_inputs = document.getElementById("fill_inputs");
        fill_inputs.showModal();

        var submit_button = document.getElementById("fill_inputs_submit");

        submit_button.addEventListener("click", function (event) {
            event.preventDefault();

            // replace input values
            var inputs = code.match(/Input\[[^\]]*\]/g);

            for (var i = 0; i < inputs.length; i++) {
                var input = inputs[i];
                // get text between ""
                var input_name = input.match(/\"[^\"]*\"/g)[0].replace(/\"/g, "");
                var input_value = document.getElementById(input_name).value;

                code = code.replace(input, `"${input_value}"`);
            }

            fill_inputs.close();
        });

        return false;
    }

    fetch(`${API_URL}/notebook`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({code:
            code,
            state_id: STATE_ID,
            is_text_cell: is_text_cell
        })
    })
    .then(response => response.json())
    .then((data) => {
        var data = data;
        var is_image = false;

        console.log(data.output, "output");

        if (data.output.image) {
            is_image = true;
            if (data.output.image.image) {
                data.output = `<img src="data:image/png;base64,${data.output.image.image}" />`;
            } else {
                data.output = `<img src="data:image/png;base64,${data.output.image}" />`;
            }
        } else if (is_text_cell) {
            data.output = DOMPurify.sanitize(marked.parse(code));
        } else if (data.output.text) {
            data.output = data.output.text;
        } else {
            data.output = "Success";
        }
        console.log(data);
        var time = data.time;
        // hide loading cell
        loading.style.display = "none";
        clearInterval(timer);
        clearInterval(output_timer);

        console.log(data, existing_cell);

        // if paper mode, take all cells and render them
        if (existing_cell == "paper_mode") {
            // update cells with all of the right cells
            // data.output will be a string delimited by \n
            
            if (data.output == "") {
                data.output = "Your program did not create an output.";
            }

            var cells = data.output.split("]");
            var notebook = document.getElementById("drag_drop_notebook");
            // set background to white
            notebook.style.background = "white";

            // remove last item
            cells.pop();

            for (var i = 0; i < cells.length; i++) {
                var function_name = cells[i].split("[")[0];
                var cell_count = i + 1;
                
                // get data-color of elemtn with id
                var color = document.getElementById(function_name).getAttribute("data-color");

                if (mapped_functions[function_name].supports_arguments) {
                    var argument_text = cells[i].split("[")[1].split("]")[0];
                    // replace " with nothing
                    argument_text = argument_text.replace(/"/g, "");
                    html = `
                        <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: 20px;">
                            <p>${function_name}[<input type="text" class="argument_block" id="cell_${cell_count}" value="${argument_text}" />]</p>
                        </div>
                    `;
                } else {
                    html = `
                        <div class="cell" draggable="true" id="${function_name}_${cell_count}" style="background-color: ${color}; margin-left: 20px;">
                            <p>${function_name}[]</p>
                        </div>
                    `;
                }

                notebook.appendChild(document.createRange().createContextualFragment(html));
            }

            return;
        }

        if (existing_cell) {
            var cell = document.getElementById(existing_cell);
            if (is_image) {
                cell.innerHTML = `
                    <p class="time">#${existing_cell} (${time}s) - <a href="#" onclick="rerun(${existing_cell})">Rerun</a></p>
                    <textarea rows="${row_count}">${code}</textarea>
                    <p>${data.output}</p>
                `;
                return;
            }
            cell.innerHTML = `
                <p class="time">#${existing_cell} (${time}s) - <a href="#" onclick="rerun(${existing_cell})">Rerun</a></p>
                <textarea rows="${row_count}">${data.output}</textarea>
                <pre ${data.error ? 'class="error_cell"' : ''}>${data.error ? data.error : data.output}</pre>
            `;
            return;
        }

        var row_count = (code.match(/\n/g) || []).length + 1;

        // if interactive mode, show in #output
        if (mode == "interactive") {
            var output = document.getElementById("output");
            output.innerHTML = data.output;
            return;
        }

        if (!is_text_cell) {
            cells.innerHTML += `
                <li class="cell" id="${cells.children.length + 1}">
                    <p class="time">#${cells.children.length + 1} (${time}s) - <a href="#" onclick="rerun(${cells.children.length + 1})">Rerun</a></p>
                    <textarea rows="${row_count}">${code}</textarea>
                    <pre ${data.error ? 'class="error_cell"' : ''}>${data.error ? data.error : data.output}</pre>
                </li>
            `;
        } else {
            cells.innerHTML += `
                <li class="cell" id="${cells.children.length + 1}">
                    <p class="time">#${cells.children.length + 1}</p>
                    ${data.output}
                </li>
            `;
        }
        
        document.getElementById("current_count").innerHTML = `#${cells.children.length + 1}`;

        // clear form text area
        var textarea = document.getElementById("jscode");
        textarea.value = "";
    })
    .catch((error) => {
        console.error('Error:', error);
        clearInterval(timer);
        clearInterval(output_timer);
        
        var error_cell = document.getElementById("error");
        error_cell.innerText = "There was an error running your code. Please make sure your code is valid.";
        error_cell.style.display = "block";
        // hide output
        output.style.display = "none";
        var loading = document.getElementById("loading");
        loading.style.display = "none";
    });
}

var form = document.getElementById("new");
var create_comment = document.getElementById("create_comment");

form.addEventListener("submit", function (event) {
    event.preventDefault();
    var data = new FormData(form);
    var code = data.get("jscode");
    executeCode(code);
});

create_comment.addEventListener("click", function (event) {
    event.preventDefault();
    var data = new FormData(form);
    var code = data.get("jscode");
    executeCode(code, comment=true);
});

// auto-expand textarea
var textarea = document.getElementById("jscode");

textarea.addEventListener("input", function (event) {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
});

function exportNotebook() {
    var export_modal = document.getElementById("export");
    export_modal.showModal();
}

function export_vic() {
    var data = new FormData();

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
}

function export_vicnb() {
    fetch(`${API_URL}/notebook/save`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({state_id: STATE_ID})
    }).then(response => response.json())
    .then(data => {
        var data = JSON.stringify(data);

        var blob = new Blob([data], {type: "text/plain;charset=utf-8"});
        blob.name = "notebook.vicnb";
        
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "notebook.vicnb";
        a.click();
    });
}

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
    if (readData && readData.startsWith("function_box")) {
        // delete element
        var element = document.getElementById(readData);
        element.parentNode.removeChild(element);
        return;
    }
    event.preventDefault();
    dropzone.style.backgroundColor = "white";

    for (var file of event.dataTransfer.files) {
        // only allow jpeg, jpg, png, or .vicnb
        file.name = file.name.toLowerCase();
        if (!file.name.endsWith(".jpg") && !file.name.endsWith(".jpeg") && !file.name.endsWith(".png") && !file.name.endsWith(".vicnb") && !file.name.endsWith(".avif") && !file.name.endsWith(".webp")) {
            var dialog = document.getElementById("dialog");
            var error_message = document.getElementById("error_message");
            error_message.innerText = "Your file could not be uploaded. Please make sure you have uploaded a supported format.";
            dialog.showModal();
            return;
        }

        uploadNotebook(event, mode, file);
    }
});

function resetNotebook() {
    var code = "Reset[]\n";
    executeCode(code);
    // show toast
    show_toast("Your notebook has been reset.");
    // delete all cells
    var cells = document.getElementById("cells");
    cells.innerHTML = "";
    // delete all interactive cells
    var notebook = document.getElementById("drag_drop_notebook");
    notebook.innerHTML = "";
    // set background back to image
    document.getElementById("drag_drop_notebook").style.background = "url('/static/img/notebook.png')";
    document.getElementById("drag_drop_notebook").style.backgroundSize = "cover";
}

function toggle_menu () {
    var menu = document.getElementById("nav_menu");
    
    if (menu.style.display == "flex") {
        menu.style.display = "none";
    } else {
        menu.style.display = "flex";
    }
}

function showFunctions(category) {
    // if all, show all
    if (category == "All") {
        var functions = document.getElementsByClassName("function");
        for (var i = 0; i < functions.length; i++) {
            var function_element = functions[i];
            console.log(function_element);
            function_element.style.display = "block";
        }
        // apply border on category sidebar
        var sidebar_item = document.getElementById("sidebar-All");
        sidebar_item.style.borderLeft = "5px solid #00ff00";

        // remove border on other categories
        var categories = ["Process", "Find", "Output", "Logic", "Deploy", "Input"];

        for (var i = 0; i < categories.length; i++) {
            var cat = categories[i];
            if (cat == category) {
                continue;
            }
            var sidebar_item = document.getElementById("sidebar-" + cat);
            sidebar_item.style.borderLeft = "none";
        }
        return;
    }

    // show .functions with data-category == category
    var functions = document.getElementsByClassName("function");
    for (var i = 0; i < functions.length; i++) {
        var function_element = functions[i];
        if (function_element.dataset.category == category) {
            function_element.style.display = "block";
            // apply green border
        } else {
            function_element.style.display = "none";
        }
    }

    // apply border on category sidebar
    var sidebar_item = document.getElementById("sidebar-" + category);
    sidebar_item.style.borderLeft = "5px solid #00ff00";

    // remove border on other categories
    var categories = ["Process", "Find", "Output", "Logic", "Deploy", "All", "Input"];

    for (var i = 0; i < categories.length; i++) {
        var cat = categories[i];
        if (cat == category) {
            continue;
        }
        var sidebar_item = document.getElementById("sidebar-" + cat);
        sidebar_item.style.borderLeft = "none";
    }

    highlighted_category = category;
}