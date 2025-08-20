async function sendMessage() {
    let input = document.getElementById("userInput").value;
    if (!input) return;
    let res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: input})
    });
    let data = await res.json();
    document.getElementById("chatbox").innerHTML += "<p><b>You:</b> " + input + "</p>";
    document.getElementById("chatbox").innerHTML += "<p><b>Bot:</b> " + data.reply + "</p>";
    document.getElementById("userInput").value = "";
}

function checkContent() {
    const content = document.getElementById("contentInput").value;
    const poster = document.getElementById("poster").value;
    const date = document.getElementById("date").value;
    const platform = document.getElementById("platform").value;

    fetch("/content-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, poster, date, platform })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("checkerResult").innerText = data.result;
    });
}


