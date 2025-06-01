// // Scroll animation for tips
// const tips = document.querySelectorAll('.tip');

// window.addEventListener('scroll', () => {
//   tips.forEach(tip => {
//     const rect = tip.getBoundingClientRect();
//     if (rect.top < window.innerHeight - 100) {
//       tip.style.opacity = '1';
//       tip.style.transform = 'translateY(0)';
//     }
//   });
// });

// // FAQ toggle
// const questions = document.querySelectorAll('.question');
// questions.forEach(btn => {
//   btn.addEventListener('click', () => {
//     const answer = btn.nextElementSibling;
//     answer.style.display = answer.style.display === 'block' ? 'none' : 'block';
//   });
// });

// Scroll animation for tips and books
const tips = document.querySelectorAll('.tip');
const books = document.querySelectorAll('.book');

function animateOnScroll(elements) {
  elements.forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight - 100) {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }
  });
}

window.addEventListener('scroll', () => {
  animateOnScroll(tips);
  animateOnScroll(books);
});

// FAQ toggle
const questions = document.querySelectorAll('.question');
questions.forEach(btn => {
  btn.addEventListener('click', () => {
    const answer = btn.nextElementSibling;
    answer.style.display = answer.style.display === 'block' ? 'none' : 'block';
  });
});
