
# Bright Data MCP Server

Bright Data Model Context Protocol (MCP) empowers your AI models and agents with real-time, reliable access to public web data. With Bright Data MCP, your applications can effortlessly retrieve both static and dynamic content from across the web, eliminating the need to build or maintain complex data scraping and unlocking infrastructure.

## Bright Data MCP Tools

These are the tools that Bright Data MCP server provides. Document them inside docs folder.:

- search_engine :	Scrape search results from Google, Bing, or Yandex. Returns SERP results in markdown format (URL, title, description).
- scrape_as_markdown :	Scrape a single webpage and return the extracted content in Markdown format. Works even on bot-protected or CAPTCHA-secured pages.
- scrape_as_html :	Same as above, but returns content in raw HTML.
- session_stats :	Provides a summary of tool usage during the current session.
- web_data_amazon_product :	Retrieve structured Amazon product data using a /dp/ URL. More reliable than scraping due to caching.
- web_data_amazon_product_reviews :	Retrieve structured Amazon review data using a /dp/ URL. Cached and reliable.
- web_data_linkedin_person_profile :	Access structured LinkedIn profile data. Cached for consistency and speed.
- web_data_linkedin_company_profile :	Access structured LinkedIn company data. Cached version improves reliability.
- web_data_zoominfo_company_profile :	Retrieve structured ZoomInfo company data. Requires a valid ZoomInfo URL.
- web_data_instagram_profiles :	Structured Instagram profile data. Requires a valid Instagram URL.
- web_data_instagram_posts :	Retrieve structured data for Instagram posts.
- web_data_instagram_reels :	Retrieve structured data for Instagram reels.
- web_data_instagram_comments :	Retrieve Instagram comments as structured data.
- web_data_facebook_posts :	Access structured data for Facebook posts.
- web_data_facebook_marketplace_listings :	Retrieve structured listings from Facebook Marketplace.
- web_data_facebook_company_reviews :	Retrieve Facebook company reviews. Requires a company URL and number of reviews.
- web_data_x_posts :	Retrieve structured data from X (formerly Twitter) posts.
- web_data_zillow_properties_listing :	Access structured Zillow listing data.
- web_data_booking_hotel_listings :	Retrieve structured hotel listings from Booking.com.
- web_data_youtube_videos :	Structured YouTube video data. Requires a valid video URL.
- scraping_browser_navigate :	Navigate the scraping browser to a new URL.
- scraping_browser_go_back :	Navigate back to the previous page.
- scraping_browser_go_forward :	Navigate forward in the browser history.
- scraping_browser_click :	Click a specific element on the page. Requires element selector.
- scraping_browser_links :	Retrieve all links on the current page along with their selectors and text.
- scraping_browser_type :	Simulate typing text into an input field.
- scraping_browser_wait_for :	Wait for a specific element to become visible.
- scraping_browser_screenshot :	Take a screenshot of the current page.
- scraping_browser_get_html :	Retrieve the full HTML of the current page. Use with care if full-page content is not needed.
- scraping_browser_get_text :	Retrieve the visible text content of the current page.