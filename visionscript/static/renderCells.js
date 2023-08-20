function uploadNotebook (event, mode, file = null) {
    var file = file || event.dataTransfer.files[0];
    var body = new FormData();
    body.append("file", file)
    body.append("state_id", STATE_ID);
    // base64 file
    var reader = new FileReader();
    // read file
    reader.readAsDataURL(file);

    // post to /notebook/upload with state id
    fetch(`${API_URL}/notebook/upload?state_id=${STATE_ID}`, {
        method: 'POST',
        body: body
    })
    .then(response => response.json())
    .then(data => {
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
                            <textarea rows="3">${code}</textarea>
                            <pre>${output}</pre>
                            <p class="rerun" onclick="rerun(${cells.children.length + 1}, '${code}')">Rerun</p>
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
            if (file_name_element.innerText == file_name) {
                return;
            }
        }

        files.innerHTML += `
            <li><img src="${base64}" alt="${file_name}" height=100 width=100 data-filename="${file_name}" style="display: block;"><span class="file_name">${file_name}</span></li>
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

        if (file.name.endsWith(".vicnb")) {
            error_message.innerText = "Please import your notebook in interactive mode.";
            dialog.showModal();
            return;
        }

        error_message.innerText = "Your file could not be uploaded. Please make sure you have uploaded a supported format.";
        dialog.showModal();
    });
}