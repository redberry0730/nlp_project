const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');

const messiUrl = 'https://api-app.espn.com/allsports/apis/v1/now?region=us&lang=en&contentorigin=espn&limit=50&contentcategories=dc5f8d51-332b-0ab2-b4b0-c97efdc624e0&offset=';
const lebronUrl = 'https://www.transfermarkt.com/lebron-james/profil/spieler/3504';

(async function scrapeESPN(url){
    const links = []
    const stories = []
    offset = 0
    while (true) {
        console.log('offset', offset)
        const response = await axios.get(url+offset);
        let feed = response.data.feed;
        if(feed.length == 0) break;
        feed.forEach((item) => {
            if (item.links.api.news)
            links.push(item.links.api.news.href);
        });    
        offset += 50;    
    }   

    for(let link of links){
        console.log('link', link)
        if (!link.includes("http://now.core.api.espn.com/v1/sports/news/")) continue;
        const response = await axios.get(link);
        const story = response.data.headlines[0].story;
        stories.push(story);
    }

    const jsonData = JSON.stringify(stories, null, 2);

    fs.writeFile('messi.json', jsonData, (err) => {
        if (err) {
            console.error('Error writing to JSON file:', err);
        } else {
            console.log('Stories saved to stories.json');
        }
    });

    console.log('stories', stories.length)
    sum = 0
    for(let story of stories){
        len = story.split(" ").length
        console.log('story', story.length)
        sum += len
    }
    console.log('avg', sum/stories.length)
    console.log("total", sum)

})(messiUrl)