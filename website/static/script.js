const text = document.querySelector(".text"); 
const url = document.querySelector(".url"); 
const form = document.querySelector("#form"); 
const switchs = document.querySelectorAll(".switch");

let current = 1; 

function tab2(){
  form.style.marginLeft = "-100%";
  text.style.background = "none";
  url.style.background = "#4d64bd"
  switchs[current-1].classList.add("active"); 
}

function tab1(){
  form.style.marginLeft = "0";
  url.style.background = "none";
  text.style.background = "#4d64bd"
  switchs[current-1].classList.remove("active"); 
}