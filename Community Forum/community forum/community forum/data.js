var defaultThreads = [
    {
        id: 1,
        title: " Spray emamectin benzoate for effective control",
        author: "Shri Haribhau",
        date: Date.now(),
        content: "Thread-con",
        comments: [
            {
                author: "Dharma Budha Dudhavde",
                date: Date.now(),
                content: "@Shri Haribhau Spray emamectin benzoate for effective control"
            },
            {
                author: "Sarathi Sahoo",
                date: Date.now(),
                content: "@Shri Haribhau Yes, you can spray Deltamethrin 2.5 EC or Cypermethrin 10 EC. I should add that it looks like Sawfly larvae that feeds on chlorophyll mostly. So, it might be different from #তুলার হেলিকোভার্পা কীড়াপোকা. Thanks for the home made organics👌🌹🌾 @Gelson Ferreira 😂💓👋"
            }
        ]
    },
    {
        id: 2,
        title: "Which manure to use for chilli crops optimum growth ?",
        author: "Shri Ravindra Ranagnath Chavan.",
        date: Date.now(),
        content: "Thread content 2",
        comments: [
            {
                author: "Ganesh Kisan Bochare",
                date: Date.now(),
                content: "Hey there"
            },
            {
                author: "Dasharath Tukaram Bochare",
                date: Date.now(),
                content: "Hey to you too"
            }
        ]
    },
    {
        id: 3,
        title: "Both are present in ewrewreweggplant what should I do? Is deltamethrin can clean these caterpilers ? They both are eating leafs of the plants what is recommended for this problem can I spray deltamethrin or cypermethrin?",
        author: "Shri Haribhau",
        date: Date.now(),
        content: "Thread content",
        comments: [
            {
                author: "Dharma Budha Dudhavde",
                date: Date.now(),
                content: "@Shri Haribhau Spray emamectin benzoate for effective control"
            },
            {
                author: "Sarathi Sahoo",
                date: Date.now(),
                content: "@Shri Haribhau Yes, you can spray Deltamethrin 2.5 EC or Cypermethrin 10 EC. I should add that it looks like Sawfly larvae that feeds on chlorophyll mostly. So, it might be different from #তুলার হেলিকোভার্পা কীড়াপোকা. Thanks for the home made organics👌🌹🌾 @Gelson Ferreira 😂💓👋"
            }
        ]
    }
]

var threads = defaultThreads
if (localStorage && localStorage.getItem('threads')) {
    threads = JSON.parse(localStorage.getItem('threads'));
} else {
    threads = defaultThreads;
    localStorage.setItem('threads', JSON.stringify(defaultThreads));
}