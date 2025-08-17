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
