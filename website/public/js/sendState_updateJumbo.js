$(document).ready(function () {
    $(".radioButton").on("change", function () {
        var data = {};
<<<<<<< HEAD
=======
        data.machine = "Machine1";
>>>>>>> 11f39fe29bb302c708c91ce1cd6b75c0e3ffdfac
        data.status = this.innerText.trim();
        data.parent_mode = $(this).parents()[2].firstChild.nextSibling.innerText.trim();
        document.getElementById("jumbotron").innerHTML = data.status;
        var col = "blue";
        if (data.parent_mode == "ON") col = "green";
        else if (data.parent_mode == "FAILED") col = "red";
        document.getElementById("jumbotron").style.color = col;
        $.ajax({
            type: 'POST',
            data: JSON.stringify(data),
            contentType: 'application/json',
            url: 'SendState',
            success: function (data) {
                console.log('success');
                console.log(JSON.stringify(data));
            }
        });
    });
});